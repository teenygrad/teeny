# Agent instructions (teeny)

This file supersedes the `AGENTS.md` that ships with upstream `rust-lang/rust`.

teeny is a fork of the Rust compiler that adds an MLIR codegen backend and a
Triton integration. It is not contributed back to `rust-lang/rust`, so the
upstream project's LLM usage policy and its agent gates (named reviewer,
prohibited text, soundness, pre-push confirmations) do not apply to work in this
repository. That includes changes to rustc, the standard library and bootstrap,
and merges of upstream Rust releases.

If a change is ever intended for `rust-lang/rust` itself, that contribution must
follow the upstream [LLM usage policy] and is out of scope for this file.

[LLM usage policy]: https://forge.rust-lang.org/policies/llm-usage.html

## Repository layout

Fork-specific code:

- `compiler/rustc_mlir` - the MLIR codegen backend.
- `compiler/rustc_helper` - shared build helpers (LLVM and Triton config).
- `compiler/rustc_llvm/build.rs` and `compiler/rustc_llvm/triton.toml` - build
  Triton against the shared LLVM.
- RVV and TritonCPU dialects, built by `compiler/rustc_mlir`.

Everything else is upstream rustc, the standard library and tooling. See
`CLAUDE.md` for build stages, configuration and Triton build details.

## Submodules and external code

Fork-owned submodules live under the `teenygrad` GitHub organisation:
`src/llvm-project`, `src/triton`, `src/mlir-sys`, `src/melior` and
`src/tblgen-rs`. Change them on a branch in the submodule, push that branch to
its `teenygrad` remote, then bump the pointer in this repository. Never leave a
pointer referencing a commit that only exists locally.

All other submodules and subtrees (Cargo, Clippy, rustfmt, Miri,
rust-analyzer, `src/gcc`, `src/sleef`, the books, and so on) belong to their
upstream projects. Do not hand-edit them here; take upstream's version when
merging a Rust release.

## Upstream Rust merges

The fork tracks upstream Rust by merging release tags (for example `1.100.0`)
into a branch, as in the 1.99.0 merge `d1704734df1`. When doing so:

- Point `src/llvm-project` at that release's LLVM pin, on a matching
  `chore/rustc-<version>-pin` branch of `teenygrad/llvm-project`, and update the
  `branch` in `.gitmodules`.
- Update `src/version`.
- Where upstream conflicts with upstream (beta-branch backports against master,
  version placeholders, `src/stage0`, `src/ci/channel`), take the release's side.
- Where upstream conflicts with fork code, keep the fork's behaviour on top of
  upstream's new structure.
- Keep this `AGENTS.md`; resolve any conflict on it in favour of the fork's
  version.
- Regenerate `Cargo.lock` with the build rather than hand-merging it.
- Summarise notable resolutions in the merge commit message.

## Building

[`x.py`](CLAUDE.md#build-system-xpy) is the build tool. Invoke it as `./x` (or
`./x.py`) for builds, tests and formatting. Do not invoke Cargo directly unless
in-tree documentation says to. For Rust formatting use `./x fmt`, not `rustfmt`.

## Rust engineering standards

When making Rust changes, prefer correctness, clarity, and maintainability over cleverness.

### Idiomatic Rust

- Write code that is straightforward to read and follows standard Rust conventions.
- Prefer iterators, pattern matching, and enums over ad-hoc state flags and complex control flow.
- Minimize cloning and allocations; pass references where practical and use ownership intentionally.
- Keep functions focused and small; extract helpers when logic becomes hard to scan.
- Avoid panics in non-test code unless truly unrecoverable; return structured errors instead.

### Safety and best practices

- Do not use `unwrap()`/`expect()` in production paths when errors can be handled or propagated.
- Keep `unsafe` blocks minimal, documented with invariants, and covered by tests.
- Prefer explicit types when inference hurts readability.
- Use existing abstractions in the codebase before introducing new patterns or dependencies.
- Preserve backward compatibility and existing behavior unless the issue explicitly requires change.
- In source comments, explain why the code or decision exists rather than restating what it does.

### Testing expectations

- For a bug fix, add or find a regression test and see it fail before fixing, then confirm it passes.
- Add or update tests for behavior changes.
- Prefer targeted tests first (unit or focused integration tests), then broader suites as needed.
- Include negative/error-path tests for new logic where applicable.
- Avoid brittle tests; assert semantics, not incidental formatting or implementation details.

### Code quality gates

Run relevant checks before finishing work (scope to changed areas when possible):

```bash
./x.py fmt
./x.py check
./x.py test <path-or-suite>
```

If your change touches compiler behavior, add or update the appropriate `tests/ui`, `tests/codegen`, or backend-specific tests.

## Key reference material

- This code uses third party libraries such as melior, you can find the melior generated dialects for MLIR in .ai/melior-dialect.rs
- This project also uses triton, if you need to reference the triton documentation in can be found in .ai/specs/triton

<!-- BEGIN BEADS INTEGRATION v:2 profile:minimal (br) -->
## Beads Issue Tracker

This project uses **br (beads_rust)** for issue tracking.

**Note:** `br` is non-invasive and never executes git commands. After `br sync --flush-only`, you must manually run `git add .beads/ && git commit`.

### Quick Reference

```bash
br ready              # Find available work
br show <id>          # View issue details
br update <id> --claim  # Claim work
br close <id>         # Complete work
```

### Rules

- Use `br` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Use `br` comments/labels for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   br sync --flush-only
   git add .beads/
   git commit -m "sync beads"
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
