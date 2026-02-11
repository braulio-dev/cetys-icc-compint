const ort = require('onnxruntime-node');

async function runModel() {
    const session = await ort.InferenceSession.create('./data/iris_pipe.onnx');

    const input = new ort.Tensor(
        "float32",
        Float32Array.from([5.1, 3.5, 1.4, 0.2]),
        [1, 4]
    );

    const feeds = {
        [session.inputNames[0]]: input
    };

    const results = await session.run(feeds);
    console.log(results);
}

runModel();
