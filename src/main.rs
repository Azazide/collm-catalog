// SPDX-License-Identifier: AGPL-3.0-only
//! `collm-catalog [--out catalog.json] [--curated curated.toml] [--force] [--refresh]`
//!
//! Exit 0: the file is written. Exit 1: the run failed and the file is untouched. Exit 2: the run
//! produced a catalog much smaller than the last one and the file is untouched (`--force` writes it).

use collm_catalog::assemble::build;
use collm_catalog::hf::HttpHf;
use collm_catalog::model::{Catalog, RULES, check_floor, check_shrink, render};
use collm_catalog::select::Curated;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const KEEP_RATIO: f64 = 0.7;

/// A run that gives fewer models than this has failed, whatever the last catalog held (a good run gives
/// about 390).
const MIN_MODELS: usize = 200;

#[derive(Debug)]
struct Args {
    out: PathBuf,
    curated: Option<PathBuf>,
    force: bool,
    refresh: bool,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args {
        out: PathBuf::from("catalog.json"),
        curated: None,
        force: false,
        refresh: false,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" => parsed.out = PathBuf::from(it.next().ok_or("--out needs a path")?),
            "--curated" => {
                parsed.curated = Some(PathBuf::from(it.next().ok_or("--curated needs a path")?))
            }
            "--force" => parsed.force = true,
            "--refresh" => parsed.refresh = true,
            "--help" | "-h" => return Err(String::new()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(parsed)
}

fn read_curated(path: Option<&Path>) -> Result<Curated, String> {
    match path {
        Some(path) => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            Curated::parse(&text)
        }
        None => match std::fs::read_to_string("curated.toml") {
            Ok(text) => Curated::parse(&text),
            Err(_) => Ok(Curated::default()),
        },
    }
}

fn write_atomically(path: &Path, text: &str) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&args) {
        Ok(args) => args,
        Err(message) => {
            if !message.is_empty() {
                eprintln!("error: {message}");
            }
            eprintln!(
                "usage: collm-catalog [--out catalog.json] [--curated curated.toml] [--force] [--refresh]"
            );
            eprintln!(
                "HF_TOKEN in the environment raises Hugging Face's rate limit; it is optional."
            );
            return if message.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            };
        }
    };
    let curated = match read_curated(args.curated.as_deref()) {
        Ok(curated) => curated,
        Err(message) => {
            eprintln!("error: {message}");
            return ExitCode::from(1);
        }
    };
    let previous = std::fs::read_to_string(&args.out)
        .map(|text| Catalog::from_text(&text))
        .unwrap_or_else(|_| Catalog::empty());
    eprintln!("last catalog: {} models", previous.models.len());

    let hf = HttpHf::new(std::env::var("HF_TOKEN").ok());
    // `--refresh` reads every repository again, and so does a catalog made under other rules (an entry
    // carried over is not redone); the shrink guard below still measures against the file.
    if !args.refresh && previous.rules != RULES {
        eprintln!(
            "the last catalog was made under rules {}, these are {RULES}: reading every repository again",
            previous.rules
        );
    }
    let carried = previous.carried(args.refresh);
    let (mut catalog, stats) = match build(&hf, &curated, &carried, &mut |line| eprintln!("{line}"))
    {
        Ok(done) => done,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };

    eprintln!(
        "listed {} (unreadable {}), looked into {}: read {}, reused {}, kept after an error {}, failed {}",
        stats.listed,
        stats.unreadable,
        stats.examined,
        stats.fetched,
        stats.reused,
        stats.kept_after_error,
        stats.failed
    );
    for (reason, count) in &stats.rejected {
        eprintln!("  left out {count}: {}", reason.text());
    }

    for (name, problems) in catalog.drop_malformed() {
        eprintln!("left out {name}: {}", problems.join("; "));
    }
    if !args.force
        && let Err(why) = check_floor(catalog.models.len(), MIN_MODELS)
    {
        eprintln!("refused: {why}");
        return ExitCode::from(2);
    }
    if !args.force
        && let Err(why) = check_shrink(previous.models.len(), catalog.models.len(), KEEP_RATIO)
    {
        eprintln!("refused: {why}");
        return ExitCode::from(2);
    }
    if let Err(e) = write_atomically(&args.out, &render(&catalog)) {
        eprintln!("error: {}: {e}", args.out.display());
        return ExitCode::from(1);
    }
    eprintln!(
        "wrote {} models to {}",
        catalog.models.len(),
        args.out.display()
    );
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|a| a.to_string()).collect()
    }

    #[test]
    fn the_defaults_and_each_flag() {
        let d = parse_args(&args(&[])).unwrap();
        assert_eq!(d.out, PathBuf::from("catalog.json"));
        assert!(d.curated.is_none() && !d.force && !d.refresh);
        let all = parse_args(&args(&[
            "--out",
            "x.json",
            "--curated",
            "c.toml",
            "--force",
            "--refresh",
        ]))
        .unwrap();
        assert_eq!(all.out, PathBuf::from("x.json"));
        assert_eq!(all.curated, Some(PathBuf::from("c.toml")));
        assert!(all.force && all.refresh);
    }

    #[test]
    fn a_missing_value_an_unknown_flag_and_help_are_errors_that_say_so() {
        assert!(parse_args(&args(&["--out"])).unwrap_err().contains("--out"));
        assert!(
            parse_args(&args(&["--bogus"]))
                .unwrap_err()
                .contains("--bogus")
        );
        assert_eq!(
            parse_args(&args(&["--help"])).unwrap_err(),
            "",
            "help is not a failure"
        );
    }
}
