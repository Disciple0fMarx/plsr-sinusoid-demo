#[cfg(feature = "plot")]
use plotters::prelude::*;

pub fn plot_results(x: &Array2<f64>, y: &Array2<f64>, plsr: &super::pls::Plsr, truth: &super::data::Truth, n: usize, p: usize, q: usize) {
    let root = BitMapBackend::new("plsr_demo.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let areas = root.split_evenly((2, 2));

    // Plot 1: True vs Predicted future (3 components)
    let y_pred = plsr.predict_with_components(y, 3);
    let mut chart = ChartBuilder::on(&areas[0])
        .caption("True vs Predicted Future", ("sans-serif", 20))
        .build_cartesian_2d(0f64..q as f64, -3.0..3.0).unwrap();
    chart.draw_series(LineSeries::new((0..q).map(|j| (j as f64, y[[0, j]])), &RED)).unwrap();
    chart.draw_series(LineSeries::new((0..q).map(|j| (j as f64, y_pred[[0, j]])), &BLUE)).unwrap();

    // More plots: scores vs true traits...
    // (you can expand this)
}

