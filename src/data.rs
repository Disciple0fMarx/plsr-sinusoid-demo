use ndarray::{Array2, Array1};
use ndarray_rand::{RandomExt, rand_distr::{Uniform, Normal}};
use rand::Rng;


#[derive(Debug)]
pub enum Case { FixedFreqPhase, RandomAFP }


pub struct Truth {
    pub amplitudes: Array1<f64>,
    pub frequencies: Array1<f64>,
    pub phases: Array1<f64>,
}


pub fn generate_sinusoids(
    n: usize,
    p: usize,
    q: usize,
    case: Case,
) -> (Array2<f64>, Array2<f64>, Truth) {
    let mut rng = rand::thread_rng();

    let amplitudes = Array1::random_using(n, Uniform::new(0.5, 3.0), &mut rng);
    let frequencies = match case {
        Case::FixedFreqPhase => Array1::from_elem(n, 1.0),
        Case::RandomAFP => Array1::random_using(n, Uniform::new(0.5, 2.0), &mut rng),
    };
    let phases = match case {
        Case::FixedFreqPhase => Array1::zeros(n),
        Case::RandomAFP => Array1::random_using(n, Uniform::new(0.0, 2.0 * std::f64::consts::PI), &mut rng),
    };

    let t_past = Array1::linspace(0.0, (p-1) as f64, q);
    let t_future = Array1::linspace(p as f64, (p + q - 1) as f64, q);

    let mut x = Array2::zeros((n, p));
    let mut y = Array2::zeros((n, q));

    for i in 0..n {
        let a = amplitudes[i];
        let f = frequencies[i];
        let phi = phases[i];

        for (j, &t) in t_past.iter().enumerate() {
            x[[i, j]] = a * (2.0 * std::f64::consts::PI * f * t * phi).sin();
        }
        for (j, &t) in t_future.iter().enumerate() {
            y[[i, j]] = a * (2.0 * std::f64::consts::PI * f * t * phi).sin();
        }
    }

    let truth = Truth { amplitudes, frequencies, phases };
    (x, y, truth)
}

