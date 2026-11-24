use std::ffi::OsStr;

use super::version_binary_name;

#[test]
fn version_names_teenyc_when_run_as_teenyc() {
    assert_eq!(version_binary_name(Some(OsStr::new("teenyc")), false), "teenyc");
    assert_eq!(version_binary_name(Some(OsStr::new("/opt/toolchain/bin/teenyc")), false), "teenyc");
    assert_eq!(version_binary_name(Some(OsStr::new("teenyc.exe")), false), "teenyc");
}

#[test]
fn version_names_rustc_for_the_rustc_copy() {
    // cargo and build scripts run the toolchain's `rustc`, and some (`rustversion`'s) reject a
    // version line that doesn't start with `rustc`.
    assert_eq!(version_binary_name(Some(OsStr::new("/opt/toolchain/bin/rustc")), false), "rustc");
    assert_eq!(version_binary_name(Some(OsStr::new("rustc.exe")), false), "rustc");
}

#[test]
fn version_names_rustc_under_bootstrap() {
    assert_eq!(version_binary_name(Some(OsStr::new("/stage1/bin/teenyc")), true), "rustc");
}

#[test]
fn version_names_rustc_for_other_or_missing_names() {
    assert_eq!(version_binary_name(Some(OsStr::new("teenyc-wrapper")), false), "rustc");
    assert_eq!(version_binary_name(Some(OsStr::new("clippy-driver")), false), "rustc");
    assert_eq!(version_binary_name(None, false), "rustc");
}
