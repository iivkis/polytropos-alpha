//! Cargo subcommand dispatches consumer-owned certified entrypoints.
//! Build semantics stay in Model::certify, never in this launcher.
use std::env;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const USAGE: &str = "usage: cargo verineural <build|verify> --certified [--manifest-path path/to/Cargo.toml] [--artifact-root path] [--pinned-sha256 64-hex-digest (verify only)]";

#[derive(Debug, PartialEq, Eq)]
struct CliError(String);
impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for CliError {}

#[derive(Debug, PartialEq, Eq)]
struct Cli {
    verify: bool,
    manifest: PathBuf,
    artifact_root: Option<PathBuf>,
    pinned: Option<String>,
}

fn fail(message: &str) -> CliError {
    CliError(format!("{message}\n{USAGE}"))
}

fn parse(args: impl IntoIterator<Item = String>, cwd: &Path) -> Result<Cli, CliError> {
    let mut args = args.into_iter().peekable();
    if args.peek().is_some_and(|a| a == "verineural") {
        args.next();
    }
    let verify = match args.next().as_deref() {
        Some("build") => false,
        Some("verify") => true,
        Some("--help" | "-h") => return Err(CliError(USAGE.into())),
        _ => return Err(fail("expected build or verify")),
    };
    let mut certified = false;
    let mut manifest: Option<PathBuf> = None;
    let mut artifact_root = None;
    let mut pinned = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--certified" if !certified => certified = true,
            "--manifest-path" if manifest.is_none() => {
                manifest = Some(PathBuf::from(
                    args.next().ok_or_else(|| fail("missing manifest path"))?,
                ));
            }
            "--artifact-root" if artifact_root.is_none() => {
                artifact_root = Some(PathBuf::from(
                    args.next().ok_or_else(|| fail("missing artifact root"))?,
                ));
            }
            "--pinned-sha256" if verify && pinned.is_none() => {
                let raw = args.next().ok_or_else(|| fail("missing pinned sha256"))?;
                if raw.len() != 64 || !raw.bytes().all(|c| c.is_ascii_hexdigit()) {
                    return Err(fail(
                        "pinned SHA256 must be exactly 64 hexadecimal characters",
                    ));
                }
                pinned = Some(raw);
            }
            _ => return Err(fail("unsupported, repeated or misplaced option")),
        }
    }
    if !certified {
        return Err(fail("--certified is required"));
    }
    let manifest = manifest.unwrap_or_else(|| PathBuf::from("Cargo.toml"));
    if manifest.file_name().is_none_or(|n| n != "Cargo.toml") {
        return Err(fail("--manifest-path must name Cargo.toml"));
    }
    let manifest = if manifest.is_absolute() {
        manifest
    } else {
        cwd.join(manifest)
    };
    Ok(Cli {
        verify,
        manifest,
        artifact_root,
        pinned,
    })
}

// Cargo's workspace example binaries share one target directory. Select the
// consumer-specific `<name>_certified.rs` target without hardcoding a lab.
fn certified_example(package: &Path) -> Result<String, CliError> {
    let entries = std::fs::read_dir(package.join("examples"))
        .map_err(|error| fail(&format!("cannot list consumer examples: {error}")))?;
    let mut matches = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| fail(&format!("cannot read example: {error}")))?;
        if !entry
            .file_type()
            .map_err(|error| fail(&format!("cannot stat example: {error}")))?
            .is_file()
        {
            continue;
        }
        let name = entry.file_name();
        let Some(stem) = name
            .to_str()
            .and_then(|name| name.strip_suffix("_certified.rs"))
        else {
            continue;
        };
        if !stem.is_empty() && stem.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
            matches.push(format!("{stem}_certified"));
        }
    }
    matches.sort();
    match matches.as_slice() {
        [only] => Ok(only.clone()),
        [] => Err(fail(
            "consumer must provide exactly one examples/<name>_certified.rs",
        )),
        _ => Err(fail(
            "multiple certified entrypoints; select a package with exactly one",
        )),
    }
}

fn run() -> Result<ExitCode, Box<dyn Error>> {
    let opts = parse(env::args().skip(1), &env::current_dir()?)?;
    let manifest = opts.manifest.canonicalize().map_err(|e| {
        CliError(format!(
            "cannot read package manifest {}: {e}",
            opts.manifest.display()
        ))
    })?;
    let package = manifest
        .parent()
        .ok_or_else(|| fail("manifest has no package directory"))?;
    let manifest_text = std::fs::read_to_string(&manifest)?;
    if !manifest_text.lines().any(|line| line.trim() == "[package]") {
        return Err(fail("select a Cargo package, not the workspace root").into());
    }
    if !package.join("src/lib.rs").is_file() {
        return Err(fail("package must define an architecture in src/lib.rs").into());
    }
    let example = certified_example(package)?;
    let mut cargo = Command::new(env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    cargo
        .arg("run")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--example")
        .arg(example)
        .arg("--");
    if opts.verify {
        cargo.arg("--verify");
    }
    if let Some(root) = opts.artifact_root {
        cargo.arg("--artifact-root").arg(root);
    }
    if let Some(pin) = opts.pinned {
        cargo.arg("--pinned-sha256").arg(pin);
    }
    let status = cargo.status()?;
    Ok(ExitCode::from(
        status.code().unwrap_or(1).try_into().unwrap_or(1),
    ))
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("verineural: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|v| (*v).to_owned()).collect()
    }

    #[test]
    fn cargo_subcommand_and_manifest_paths() {
        let cli = parse(
            strings(&[
                "verineural",
                "build",
                "--certified",
                "--manifest-path",
                "labs/x/Cargo.toml",
                "--artifact-root",
                "out",
            ]),
            Path::new("/repo"),
        )
        .unwrap();
        assert_eq!(cli.manifest, Path::new("/repo/labs/x/Cargo.toml"));
        assert_eq!(cli.artifact_root.as_deref(), Some(Path::new("out")));
        assert!(!cli.verify);
    }

    #[test]
    fn certified_entrypoint_is_unique_and_architecture_neutral() {
        let root = std::env::temp_dir().join(format!(
            "verineural-example-selection-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("examples")).unwrap();
        assert!(certified_example(&root).is_err());
        std::fs::write(root.join("examples/xor_certified.rs"), "").unwrap();
        std::fs::write(root.join("examples/other.rs"), "").unwrap();
        assert_eq!(certified_example(&root).unwrap(), "xor_certified");
        std::fs::write(root.join("examples/other_certified.rs"), "").unwrap();
        assert!(certified_example(&root).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn no_implicit_certification_or_invalid_pin() {
        assert!(parse(strings(&["build"]), Path::new("/tmp")).is_err());
        assert!(
            parse(
                strings(&["build", "--certified", "--pinned-sha256", "00"]),
                Path::new("/tmp")
            )
            .is_err()
        );
        assert!(
            parse(
                strings(&["verify", "--certified", "--pinned-sha256", "00"]),
                Path::new("/tmp")
            )
            .is_err()
        );
        let pin = "a".repeat(64);
        assert!(
            parse(
                vec![
                    "verify".into(),
                    "--certified".into(),
                    "--pinned-sha256".into(),
                    pin
                ],
                Path::new("/tmp")
            )
            .is_ok()
        );
    }
}
