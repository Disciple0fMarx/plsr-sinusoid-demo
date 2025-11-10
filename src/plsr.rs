use ndarray::{Array2, Array1, Axis, s};
use ndarray_linalg::SVDSquare;


pub struct Plsr {
    x_mean: Array1<f64>,
    y_mean: Array1<f64>,
    w: Array2<f64>,       // Weights
    p: Array2<f64>,       // X loadings
    q: Array2<f64>,       // Y loadings
    t: Array2<f64>,       // Scores (full)
}


impl Plsr {
    pub fn fit(x: &Array2<f64>, y: &Array2<f64>, max_comp: usize) -> Self {
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let y_mean = y.mean_axis(Axis(0)).unwrap();
        let xc = x - &x_mean;
        let yc = y - &y_mean;

        let mut w = Array2::zeros((x.ncols(), max_comp));
        let mut p = Array2::zeros((x.ncols(), max_comp));
        let mut q = Array2::zeros((y.ncols(), max_comp));
        let mut t = Array2::zeros((x.nrows(), max_comp));

        let mut s = xc.t().dot(&yc);  // Cross-covariance

        for a in 0..max_comp {
            let svd = s.svd_square(true, true).unwrap();
            let w_a = svd.0.column(0).to_owned();
            let q_a = svd.2.column(0).to_owned();

            let t_a = xc.dot(&w_a);
            let p_a = xc.t().dot(&t_a) / t_a.dot(&t_a);
            let v_a = if a == 0 { p_a.clone() } else { p_a - p.slice(s![.., ..a]).dot(&p.slice(s![.., ..a]).t().dot(&p_a)) };

            s = s - &v_a.dot(&v_a.t().dot(&s));

            w.column_mut(a).assign(&w_a);
            p.column_mut(a).assign(&p_a);
            q.column_mut(a).assign(&q_a);
            t.column_mut(a).assign(&t_a);
        }

        Plsr { x_mean, y_mean, w, p, q, t }
    }

    pub fn predict_with_components(&self, x: &Array2<f64>, n_comp: usize) -> Array2<f64> {
        let xc = x - &self.x_mean;
        let t_new = xc.dot(&self.w.slice(s![.., ..n_comp]).t());
        let y_pred_c = t_new.dot(&self.q.slice(s![.., ..n_comp]).t());
        &y_pred_c + &self.y_mean
    }

    pub fn scores(&self, n_comp: usize) -> Array2<f64> {
        self.t.slice(s![.., ..n_comp]).to_owned()
    }
}

