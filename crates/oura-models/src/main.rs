use anyhow::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (sub, rest) = args.split_first().map(|(a, b)| (a.as_str(), b.to_vec())).unwrap_or(("", vec![]));
    match sub {
        "cva" => oura_models::cva::run(&rest),
        "activity" => oura_models::activity::run(&rest),
        "illness" => oura_models::illness::run(&rest),
        "stress" => oura_models::stress::run(&rest),
        _ => {
            eprintln!("usage: oura-models <cva|activity|illness|stress> [args…] (see plan doc)");
            Ok(())
        }
    }
}
