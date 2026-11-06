use burn::tensor::{Device, Tensor, TensorData, module::attention};

fn input(shape: [usize; 4], device: &Device) -> Tensor<4> {
    let values: Vec<f32> = (0..shape.iter().product())
        .map(|i| (i % 97) as f32 / 97.0)
        .collect();
    Tensor::from_data(TensorData::new(values, shape), device)
        .swap_dims(1, 2)
        .require_grad()
}

fn main() {
    let dims: Vec<usize> = std::env::args()
        .skip(1)
        .map(|x| x.parse().expect("dimensions must be integers"))
        .collect();
    let [batch, heads, queries, keys, dim, value_dim]: [usize; 6] = dims
        .try_into()
        .expect("usage: attention-autotune-repro B H Q K D DV");
    let device = Device::cuda(0).autodiff();
    let q = input([batch, queries, heads, dim], &device);
    let k = input([batch, keys, heads, dim], &device);
    let v = input([batch, keys, heads, value_dim], &device);
    println!(
        "attention: Q={:?}, K={:?}, V={:?}",
        q.dims(),
        k.dims(),
        v.dims()
    );
    let output = attention(
        q.clone(),
        k.clone(),
        v.clone(),
        None,
        None,
        Default::default(),
    );
    let loss = output.sum();
    let scalar = loss.clone().into_scalar::<f32>();
    assert!(scalar.is_finite());
    println!("forward passed: sum={scalar}");
    let gradients = loss.backward();
    for (name, input) in [("Q", q), ("K", k), ("V", v)] {
        let gradient = input.grad(&gradients).expect("gradient must exist");
        let sum = gradient.sum().into_scalar::<f32>();
        assert!(sum.is_finite());
        println!("backward passed: {name} gradient sum={sum}");
    }
}
