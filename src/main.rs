mod data;
mod plsr;
mod metrics;
// mod viz;


use data::{generate_sinusoids, Case};
use metrics::{mse, nmse, r2_y, correlation};
use plsr::Plsr;
use ndarray::Array2;


fn main() {
    let n = 300;
    let p = 100;
    let q = 30;
    let max_components = 5;

    // Generate data with 3 hidden traits
    let (x, y, truth) = generate_sinusoids(n, p, q, Case::RandomAFP);

    println!("PLSR Sinusoid Demo\n");
    println!("n = {}, p = {}, q = {}", n, p, q);

    let plsr = Plsr::fit(&x, &y, max_components);

    // Evaluate each number of components
    println!("\n{:>4} {:>12} {:>10} {:>10} {:>10}", "Comp", "MSE", "NMSE", "R²_Y", "Corr");
    for a in 1..=max_components {
        let y_pred = plsr.predict_with_components(&x, a);
        let mse_val = mse(&y, &y_pred);
        let nmse_val = nmse(&y, &y_pred);
        let r2 = r2_y(&y, &y_pred);

        // Correlation with true latent variables
        let scores = plsr.scores(a);
        let corr_a = correlation(&scores.column(0), &truth.amplitudes);
        let corr_f = if a >= 2 { correlation(&scores.column(1), &truth.frequencies) } else { 0.0 };
        let corr_p = if a >= 3 { correlation(&scores.column(2), &truth.phases) } else { 0.0 };

        println!(
            "{:4} {:12.6} {:10.2e} {:10.6} {:10.3} {:6.3} {:6.3}",
            a, mse_val, nmse_val, r2, corr_a, corr_f, corr_p
        );
    }

    // Optional: plot results
    // #[cfg(feature = "plot")]
    // viz::plot_results(&x, &y, &plsr, &truth, n, p, q);
}

