use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::PathBuf;

use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "comet-tool", about = "Comet command-line tools")]
enum Opt {
    /// Utility to extract detailed native metrics from Spark executor's
    /// stderr log when `spark.comet.explain.native.enabled=true`.
    ExtractNativeMetrics {
        /// Path to stderr output from Spark executor
        #[structopt(parse(from_os_str))]
        path: PathBuf,
    },
}

#[derive(Debug, Default)]
struct Stats {
    /// Number of times this executor appears in the log
    count: usize,
    /// Total elapsed compute time
    time_nanos: f64,
}

fn main() -> io::Result<()> {
    let opt = Opt::from_args();
    match opt {
        Opt::ExtractNativeMetrics { path } => extract_native_metrics(&path),
    }
}

fn extract_native_metrics(path: &PathBuf) -> io::Result<()> {
    let file = File::open(&path)?;
    let reader = io::BufReader::new(file);

    let mut map: HashMap<String, Stats> = HashMap::new();
    let re =
        Regex::new(r"^\s*(.*Exec|SortMergeJoin).*(elapsed_compute|join_time)=([0-9.]*)([a-zµ]*)")
            .unwrap();

    for line in reader.lines() {
        let line = line?;
        if let Some(captures) = re.captures(&line) {
            let operator = captures.get(1).unwrap().as_str();
            let time = captures.get(3).unwrap().as_str().parse::<f64>().unwrap();
            let unit = captures.get(4).unwrap().as_str();
            if let Some(time_unit) = parse_time_unit(&line, unit) {
                let unique_exec_str: String = match operator {
                    "ScanExec" => format!(
                        "{operator}: {}",
                        extract_between(&line, "source=", ", metrics").unwrap_or("")
                    ),
                    "FilterExec" => format!(
                        "{operator}: {}",
                        extract_between(&line, " ", ", metrics").unwrap_or("")
                    ),
                    "SortExec" => format!(
                        "{operator}: {}",
                        extract_between(&line, "expr=[", "]").unwrap_or("")
                    ),
                    "ShuffleWriterExec" => format!(
                        "{operator}: {}",
                        extract_between(&line, "partitioning=", "}]").unwrap_or("")
                    ),
                    "AggregateExec" => format!(
                        "{operator}: {}",
                        extract_between(&line, "mode=", ", metrics").unwrap_or("")
                    ),
                    "HashJoinExec" | "SortMergeJoin" => format!(
                        "{operator}: {}",
                        extract_between(&line, "join_type=", "]").unwrap_or("")
                    ),
                    _ => operator.to_string(),
                };
                let stats = map.entry(unique_exec_str).or_insert(Stats::default());
                stats.count += 1;
                stats.time_nanos += time * time_unit;
            }
        }
    }

    for (k, v) in &map {
        println!("{}\t{}\t{k}", v.count, (v.time_nanos / 1000000.0) as i64);
    }

    Ok(())
}

fn extract_between<'a>(input: &'a str, start: &'a str, end: &'a str) -> Option<&'a str> {
    if let (Some(start), Some(end)) = (input.find(start), input.find(end)) {
        if start < end {
            return Some(&input[start..=end]);
        }
    }
    None
}

fn parse_time_unit(line: &str, unit: &str) -> Option<f64> {
    match unit {
        "s" => Some(1000000000.0),
        "ms" => Some(1000000.0),
        "µs" => Some(1000.0),
        "ns" => Some(1.0),
        _ => {
            eprintln!("unknown unit '{unit}' in {line}");
            None
        }
    }
}
