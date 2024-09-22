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
    let re = Regex::new(r"^\s*(.*Exec).*elapsed_compute=([0-9.]*)([a-zµ]*)").unwrap();

    for line in reader.lines() {
        let line = line?;
        if let Some(captures) = re.captures(&line) {
            let operator = captures.get(1).unwrap().as_str();
            let time = captures.get(2).unwrap().as_str().parse::<f64>().unwrap();
            let unit = captures.get(3).unwrap().as_str();
            let time_unit = match unit {
                "s" => 1000000000.0,
                "ms" => 1000000.0,
                "µs" => 1000.0,
                "ns" => 1.0,
                _ => unreachable!("unknown unit: {unit}"),
            };
            let stats = map.entry(operator.to_string()).or_insert(Stats::default());
            stats.count += 1;
            stats.time_nanos += time * time_unit;
        }
    }

    for (k, v) in &map {
        println!("{k},{},{} ms", v.count, (v.time_nanos / 1000000.0) as i64);
    }

    Ok(())
}
