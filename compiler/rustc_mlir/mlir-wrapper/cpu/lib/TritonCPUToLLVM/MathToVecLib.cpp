#include "TypeConverter.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_MATHTOVECLIB
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

template <typename OpT> struct VecOpToFp32 : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;

  VecOpToFp32(MLIRContext *context) : OpRewritePattern<OpT>(context) {}

  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy)
      return failure();

    Type elemTy = vecTy.getElementType();
    if (!elemTy.isBF16() && !elemTy.isF16())
      return failure();

    Type fp32VecTy = vecTy.cloneWith(std::nullopt, rewriter.getF32Type());
    SmallVector<Value> fp32Ops;
    for (auto operand : op->getOperands())
      fp32Ops.push_back(
          arith::ExtFOp::create(rewriter, loc, fp32VecTy, operand));
    auto newOp = OpT::create(rewriter, loc, fp32VecTy, fp32Ops);
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, vecTy, newOp);
    return success();
  }
};

// Decompose vector operation to single-dimensional vector operations
// with a AVX512 for x86 or NEON for ARM.
template <typename OpT>
struct DecomposeToNativeVecs : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;
  // CPU SIMD vector size in bits
  size_t vec_bits;

  DecomposeToNativeVecs(MLIRContext *context,
                        size_t native_vec_size_in_bits = 512)
      : OpRewritePattern<OpT>(context), vec_bits(native_vec_size_in_bits) {}

  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy)
      return failure();

    Type elemTy = vecTy.getElementType();
    if (!elemTy.isF32() && !elemTy.isF64())
      return failure();

    int64_t numElems = vecTy.getNumElements();
    if (numElems * elemTy.getIntOrFloatBitWidth() < 128)
      return failure();

    // Produce a new shape where trailing dimensions wouldn't exceed the native
    // vector size.
    auto shape = vecTy.getShape();
    SmallVector<int64_t> newShape(1, 1);
    int64_t elemsPerVec = vec_bits / elemTy.getIntOrFloatBitWidth();
    for (int64_t i = shape.size() - 1; i >= 0; --i) {
      int64_t size = shape[i];
      if (newShape.size() > 1) {
        newShape.insert(newShape.begin(), size);
      } else {
        int64_t combined = newShape[0] * size;
        if (combined > elemsPerVec) {
          newShape[0] = elemsPerVec;
          newShape.insert(newShape.begin(), combined / elemsPerVec);
        } else {
          newShape[0] = combined;
        }
      }
    }
    if (newShape == shape)
      return failure();

    // Convert input operand to the new shape.
    SmallVector<Value> reshapedInputs;
    for (auto operand : op->getOperands()) {
      auto operandTy = cast<VectorType>(operand.getType());
      auto newOperandTy = VectorType::get(newShape, operandTy.getElementType());
      reshapedInputs.push_back(
          vector::ShapeCastOp::create(rewriter, loc, newOperandTy, operand));
    }

    // Decompose the original operation to a set of operations on native
    // vectors.
    auto newOpTy = VectorType::get(newShape, elemTy);
    auto subResTy = VectorType::get(newShape.back(), elemTy);
    Value newRes = arith::ConstantOp::create(
        rewriter, loc,
        SplatElementsAttr::get(newOpTy, rewriter.getFloatAttr(elemTy, 0)));
    auto strides = computeStrides(newShape);
    // Remove the last stride to produce sub-vector indices.
    strides.pop_back();
    for (int64_t idx = 0; idx < numElems; idx += newShape.back()) {
      auto indices = delinearize(idx, strides);
      SmallVector<Value> subInputs(reshapedInputs.size());
      std::transform(reshapedInputs.begin(), reshapedInputs.end(),
                     subInputs.begin(), [&](auto val) {
                       return vector::ExtractOp::create(rewriter, loc, val,
                                                        indices);
                     });
      Value subRes =
          OpT::create(rewriter, loc, subResTy, subInputs, op->getAttrs());
      newRes = vector::InsertOp::create(rewriter, loc, subRes, newRes, indices);
    }

    // Reshape the result back to the original type.
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, vecTy, newRes);
    return success();
  }
};

using ExternElementwiseOp = triton::cpu::ExternElementwiseOp;

/*
 * libsleef does not contain implementations for 2-element vectors, so we pad
 * any such vectors to size 4 instead.
 */
struct PadSmallVecsForSleef : public OpRewritePattern<ExternElementwiseOp> {
public:
  using OpRewritePattern<ExternElementwiseOp>::OpRewritePattern;

  PadSmallVecsForSleef(MLIRContext *context)
      : OpRewritePattern<ExternElementwiseOp>(context) {}

  LogicalResult matchAndRewrite(ExternElementwiseOp op,
                                PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy)
      return failure();

    Type elemTy = vecTy.getElementType();
    if (!elemTy.isF32() && !elemTy.isF64())
      return failure();

    int64_t numElems = vecTy.getNumElements();
    if (numElems >= 4)
      return failure();

    // Create a single-element vector for shuffle to use
    auto paddingVec = vector::BroadcastOp::create(
        rewriter, loc, VectorType::get({1}, elemTy), b.undef(elemTy));
    // Assign indices such that shuffle will pad the original vector with
    // elements from the paddingVec
    SmallVector<int64_t> indices(4);
    for (int i = 0; i < 4; ++i) {
      if (i < numElems)
        indices[i] = i;
      else
        indices[i] = numElems;
    }
    SmallVector<Value> newOperands;
    for (auto argVal : op.getOperands()) {
      auto shuf =
          vector::ShuffleOp::create(rewriter, loc, argVal, paddingVec, indices);
      newOperands.push_back(shuf.getResult());
    }
    // Update return type of extern call
    auto newVecTy = VectorType::get({4}, elemTy);
    auto extern_elem = ExternElementwiseOp::create(
        rewriter, loc, newVecTy, newOperands, op.getSymbol(), op.getPure());
    indices.resize(numElems);
    // Truncate result to original size
    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(op, extern_elem.getResult(),
                                                   paddingVec, indices);
    return success();
  }
};

using GetVecFnNameFn = std::function<std::string(
    unsigned /*bitwidth*/, unsigned /*numel*/, ValueRange /*operands*/)>;

class MvecNameGenerator {
public:
  explicit MvecNameGenerator(StringRef baseName) : baseName(baseName) {}

  std::string operator()(unsigned bitwidth, unsigned numel,
                         ValueRange operands) const {
    if (bitwidth != 32 && bitwidth != 64)
      return "";
    unsigned vecSize = numel * bitwidth;
    std::string isaPrefix;
    if (vecSize == 128) {
      isaPrefix = "b";
    } else if (vecSize == 256) {
      isaPrefix = "d";
    } else if (vecSize == 512) {
      isaPrefix = "e";
    } else {
      return "";
    }
    std::string fnName = "_ZGV" + isaPrefix + "N" + std::to_string(numel);
    for (auto operand : operands)
      fnName += "v";
    return fnName + "_" + baseName + (bitwidth == 32 ? "f" : "");
  }

private:
  std::string baseName;
};

std::string sleefUlpSuffix(unsigned ulp) {
  if (ulp == 0)
    return "";
  char buf[5]; // 4 char suffix + '\0' added by snprintf
  snprintf(buf, 5, "_u%02u", ulp);
  return buf;
}

class SleefNameGenerator {
public:
  SleefNameGenerator(StringRef baseName, unsigned ulp = 10)
      : baseName(baseName), ulpSuffix(sleefUlpSuffix(ulp)) {}

  std::string operator()(unsigned bitwidth, unsigned numel,
                         ValueRange /*operands*/) const {
    if (bitwidth != 32 && bitwidth != 64)
      return "";
    unsigned vecSize = numel * bitwidth;
    if (vecSize < 128)
      return "";
    return "Sleef_" + baseName + (bitwidth == 32 ? "f" : "d") +
           std::to_string(numel) + ulpSuffix;
  }

private:
  std::string baseName;
  std::string ulpSuffix;
};

/*
 * Names libsleef's RVV functions, e.g. Sleef_sinfx_u10rvvm2: "x" marks a
 * scalable vector, and rvvm<N> the LMUL whose minimum size (VLEN = 128) the
 * fixed vector fills. libsleef only provides LMUL 1 and 2.
 */
class RvvSleefNameGenerator {
public:
  RvvSleefNameGenerator(StringRef baseName, unsigned ulp = 10)
      : baseName(baseName), ulpSuffix(sleefUlpSuffix(ulp)) {}

  std::string operator()(unsigned bitwidth, unsigned numel,
                         ValueRange /*operands*/) const {
    if (bitwidth != 32 && bitwidth != 64)
      return "";
    unsigned vecSize = numel * bitwidth;
    if (vecSize != 128 && vecSize != 256)
      return "";
    return "Sleef_" + baseName + (bitwidth == 32 ? "f" : "d") + "x" +
           ulpSuffix + "rvvm" + std::to_string(vecSize / 128);
  }

private:
  std::string baseName;
  std::string ulpSuffix;
};

template <typename OpT>
struct OpToVecLibConversion : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;

  virtual std::string getVecFnName(OpT op, unsigned bitwidth,
                                   unsigned numel) const = 0;

  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const {
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy || vecTy.getRank() > 1)
      return failure();

    auto fnName = getVecFnName(op, vecTy.getElementTypeBitWidth(),
                               vecTy.getNumElements());
    if (fnName.empty())
      return failure();

    auto module = SymbolTable::getNearestSymbolTable(op);
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, fnName));
    // Generate function declaration if it doesn't exists yet.
    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());
      auto fnTy = FunctionType::get(
          rewriter.getContext(), op->getOperandTypes(), op->getResultTypes());
      opFunc = func::FuncOp::create(rewriter, rewriter.getUnknownLoc(), fnName,
                                    fnTy);
      opFunc.setPrivate();
      opFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
                      UnitAttr::get(rewriter.getContext()));
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, fnName, op.getType(),
                                              op->getOperands());
    return success();
  }
};

template <typename OpT>
struct VecOpToVecLibConversion : public OpToVecLibConversion<OpT> {
public:
  VecOpToVecLibConversion(MLIRContext *context, GetVecFnNameFn getVecFnName)
      : OpToVecLibConversion<OpT>(context), getVecFnNameImpl(getVecFnName) {}

  std::string getVecFnName(OpT op, unsigned bitwidth,
                           unsigned numel) const override {
    return getVecFnNameImpl(bitwidth, numel, op->getOperands());
  }

private:
  GetVecFnNameFn getVecFnNameImpl;
};

/*
 * libsleef's RVV functions take and return scalable vectors (vfloat32m2_t
 * etc.), which the RVV calling convention passes in vector registers, where a
 * fixed-width vector would be passed in memory. So each fixed vector is
 * inserted at the start of the scalable type of the same minimum size, and
 * the result extracted back out of it. The V extension guarantees VLEN >= 128,
 * so the scalable vector always has room for the whole fixed one; the lanes
 * past it are computed on zeros and dropped.
 */
template <typename OpT>
struct RvvVecOpToVecLibConversion : public OpRewritePattern<OpT> {
public:
  RvvVecOpToVecLibConversion(MLIRContext *context, GetVecFnNameFn getVecFnName)
      : OpRewritePattern<OpT>(context), getVecFnName(getVecFnName) {}

  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy || vecTy.getRank() > 1 || vecTy.isScalable())
      return failure();

    int64_t numElems = vecTy.getNumElements();
    auto fnName = getVecFnName(vecTy.getElementTypeBitWidth(), numElems,
                               op->getOperands());
    if (fnName.empty())
      return failure();

    // vscale is VLEN / 64, so at VLEN = 128 the scalable type holds twice its
    // minimum element count.
    Type elemTy = vecTy.getElementType();
    auto scalableTy = VectorType::get({numElems / 2}, elemTy, {true});

    auto module = SymbolTable::getNearestSymbolTable(op);
    if (!SymbolTable::lookupSymbolIn(module, fnName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());
      SmallVector<Type> argTys(op->getNumOperands(), scalableTy);
      auto fnTy = FunctionType::get(rewriter.getContext(), argTys, scalableTy);
      auto opFunc = func::FuncOp::create(rewriter, rewriter.getUnknownLoc(),
                                         fnName, fnTy);
      opFunc.setPrivate();
      opFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
                      UnitAttr::get(rewriter.getContext()));
    }

    Value zeros = arith::ConstantOp::create(
        rewriter, loc,
        SplatElementsAttr::get(scalableTy, rewriter.getFloatAttr(elemTy, 0)));
    SmallVector<Value> args;
    for (Value operand : op->getOperands())
      args.push_back(
          vector::ScalableInsertOp::create(rewriter, loc, operand, zeros, 0));
    auto call = func::CallOp::create(rewriter, loc, fnName,
                                     TypeRange{scalableTy}, args);
    rewriter.replaceOpWithNewOp<vector::ScalableExtractOp>(
        op, vecTy, call.getResult(0), 0);
    return success();
  }

private:
  GetVecFnNameFn getVecFnName;
};

struct ExternElementwiseOpConversion
    : public OpToVecLibConversion<triton::cpu::ExternElementwiseOp> {
  using OpToVecLibConversion::OpToVecLibConversion;

  std::string getVecFnName(triton::cpu::ExternElementwiseOp op,
                           unsigned bitwidth, unsigned numel) const override {
    auto fnName = op.getSymbol();
    auto numelIdx = fnName.find("%(numel)");
    if (numelIdx == StringRef::npos)
      return fnName.str();
    return (fnName.take_front(numelIdx) + Twine(numel) +
            fnName.drop_front(numelIdx + 8))
        .str();
  }
};

template <typename OpTy,
          template <typename> class ConversionT = VecOpToVecLibConversion>
void populatePatternsForOp(RewritePatternSet &patterns,
                           GetVecFnNameFn getVecFnName,
                           size_t vec_size_in_bits = 512) {
  patterns.add<VecOpToFp32<OpTy>>(patterns.getContext());
  patterns.add<DecomposeToNativeVecs<OpTy>>(patterns.getContext(),
                                            vec_size_in_bits);
  patterns.add<ConversionT<OpTy>>(patterns.getContext(), getVecFnName);
}

struct MathToVecLibPass
    : public mlir::triton::cpu::impl::MathToVecLibBase<MathToVecLibPass> {
  MathToVecLibPass() = default;
  // Default to 128-bit if no features are specified.
  size_t vec_size_in_bits = 128;
  // RISC-V V: calls libsleef's scalable-vector functions.
  bool rvv = false;

  explicit MathToVecLibPass(VecLib lib, std::set<std::string> cpu_features) {
    this->lib = lib;
    update_vec_size(cpu_features);
  }

  void update_vec_size(std::set<std::string> &cpu_features) {
    // TODO:
    //  Refactor this as an independent function.
    //  And improve this to support other x86 SIMD ISAs and also for arm SVE
    //  (VLA)
    for (auto feature : cpu_features) {
      if (feature == "avx512f") {
        vec_size_in_bits = std::max<size_t>(vec_size_in_bits, 512);
      } else if (feature == "avx") {
        vec_size_in_bits = std::max<size_t>(vec_size_in_bits, 256);
      } else if (feature == "sse") {
        vec_size_in_bits = std::max<size_t>(vec_size_in_bits, 128);
      } else if (feature == "neon") {
        // Arm NEON is fixed 128-bit SIMD ISA.
        vec_size_in_bits = 128;
        break;
      } else if (feature == "v") {
        // libsleef's widest RVV functions are LMUL 2: 256 bits at the
        // VLEN = 128 the V extension guarantees.
        rvv = true;
        vec_size_in_bits = 256;
        break;
      }
    }
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    RewritePatternSet patterns(context);

    if (!cpu_features.empty()) {
      std::set<std::string> cpu_features_set{cpu_features.begin(),
                                             cpu_features.end()};
      update_vec_size(cpu_features_set);
    }

    switch (lib) {
    case VecLib::Mvec: {
      populateCommonPatterns<MvecNameGenerator>(patterns);
      break;
    }
    case VecLib::Sleef: {
      if (rvv) {
        // floor, sqrt and trunc stay as math ops: RVV has instructions for
        // them, which beat a call.
        populateCommonPatterns<RvvSleefNameGenerator,
                               RvvVecOpToVecLibConversion>(patterns);
        populatePatternsForOp<math::ExpM1Op, RvvVecOpToVecLibConversion>(
            patterns, RvvSleefNameGenerator("expm1"), vec_size_in_bits);
        break;
      }
      populateCommonPatterns<SleefNameGenerator>(patterns);
      populatePatternsForOp<math::ExpM1Op>(
          patterns, SleefNameGenerator("expm1"), vec_size_in_bits);
      populatePatternsForOp<math::FloorOp>(
          patterns, SleefNameGenerator("floor", /*ulp=*/0), vec_size_in_bits);
      populatePatternsForOp<math::SqrtOp>(
          patterns, SleefNameGenerator("sqrt", /*ulp=*/5), vec_size_in_bits);
      populatePatternsForOp<math::TruncOp>(
          patterns, SleefNameGenerator("trunc", /*ulp=*/0), vec_size_in_bits);
      break;
    }
    }

    patterns.add<DecomposeToNativeVecs<ExternElementwiseOp>>(
        patterns.getContext(), vec_size_in_bits);
    patterns.add<PadSmallVecsForSleef>(patterns.getContext());
    patterns.add<ExternElementwiseOpConversion>(patterns.getContext());

    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
  }

  template <typename VecFnNameGenerator,
            template <typename> class ConversionT = VecOpToVecLibConversion>
  void populateCommonPatterns(RewritePatternSet &patterns) const {
    populatePatternsForOp<math::AcosOp, ConversionT>(
        patterns, VecFnNameGenerator("acos"), vec_size_in_bits);
    populatePatternsForOp<math::AcoshOp, ConversionT>(
        patterns, VecFnNameGenerator("acosh"), vec_size_in_bits);
    populatePatternsForOp<math::AsinOp, ConversionT>(
        patterns, VecFnNameGenerator("asin"), vec_size_in_bits);
    populatePatternsForOp<math::AsinhOp, ConversionT>(
        patterns, VecFnNameGenerator("asinh"), vec_size_in_bits);
    populatePatternsForOp<math::AtanOp, ConversionT>(
        patterns, VecFnNameGenerator("atan"), vec_size_in_bits);
    populatePatternsForOp<math::AtanhOp, ConversionT>(
        patterns, VecFnNameGenerator("atanh"), vec_size_in_bits);
    populatePatternsForOp<math::CbrtOp, ConversionT>(
        patterns, VecFnNameGenerator("cbrt"), vec_size_in_bits);
    populatePatternsForOp<math::CosOp, ConversionT>(
        patterns, VecFnNameGenerator("cos"), vec_size_in_bits);
    populatePatternsForOp<math::CoshOp, ConversionT>(
        patterns, VecFnNameGenerator("cosh"), vec_size_in_bits);
    populatePatternsForOp<math::ErfOp, ConversionT>(
        patterns, VecFnNameGenerator("erf"), vec_size_in_bits);
    populatePatternsForOp<math::ExpOp, ConversionT>(
        patterns, VecFnNameGenerator("exp"), vec_size_in_bits);
    populatePatternsForOp<math::Exp2Op, ConversionT>(
        patterns, VecFnNameGenerator("exp2"), vec_size_in_bits);
    populatePatternsForOp<math::LogOp, ConversionT>(
        patterns, VecFnNameGenerator("log"), vec_size_in_bits);
    populatePatternsForOp<math::Log2Op, ConversionT>(
        patterns, VecFnNameGenerator("log2"), vec_size_in_bits);
    populatePatternsForOp<math::Log10Op, ConversionT>(
        patterns, VecFnNameGenerator("log10"), vec_size_in_bits);
    populatePatternsForOp<math::Log1pOp, ConversionT>(
        patterns, VecFnNameGenerator("log1p"), vec_size_in_bits);
    populatePatternsForOp<math::SinOp, ConversionT>(
        patterns, VecFnNameGenerator("sin"), vec_size_in_bits);
    populatePatternsForOp<math::SinhOp, ConversionT>(
        patterns, VecFnNameGenerator("sinh"), vec_size_in_bits);
    populatePatternsForOp<math::TanOp, ConversionT>(
        patterns, VecFnNameGenerator("tan"), vec_size_in_bits);
    populatePatternsForOp<math::TanhOp, ConversionT>(
        patterns, VecFnNameGenerator("tanh"), vec_size_in_bits);
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>>
createMathToVecLibPass(VecLib lib, std::set<std::string> cpu_features) {
  return std::make_unique<MathToVecLibPass>(lib, cpu_features);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
