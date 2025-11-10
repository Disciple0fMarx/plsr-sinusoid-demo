use ndarray::{Array2, Array1, Axis};


pub fn mse(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    (y_true - y_pred).mapv(|x| x.powi(2)).mean().unwrap()
}


pub fn nmse(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    let num = (y_true - y_pred).mapv(|x| x.powi(2)).sum();
    let den = y_true.mapv(|x| (x - y_true.mean().unwrap()).powi(2)).sum();
    num / den
}


pub fn r2_y(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    1.0 - nmse(y_true, y_pred)
}


pub fn correlation(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let ma = a.mean().unwrap();
    let mb = b.mean().unwrap();
    let num = ((a - ma) * (b - mb)).sum();
    let den = ((a - ma).mapv(|x| x.powi(2)).sum() * (b - mb).mapv(|x| powi(2)).sum()).sqrt();
    num / den
}

