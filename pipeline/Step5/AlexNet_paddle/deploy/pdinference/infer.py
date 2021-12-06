import os
import paddle
from paddle import inference
import numpy as np
from PIL import Image

from reprod_log import ReprodLogger
from preprocess_ops import ResizeImage, CenterCropImage, NormalizeImage, ToCHW, Compose


def load_predictor(model_file_path, params_file_path, args):
    config = inference.Config(model_file_path, params_file_path)
    if args.use_gpu:
        config.enable_use_gpu(1000, 0)
    else:
        config.disable_gpu()
        if args.use_mkldnn:
            config.enable_mkldnn()
            config.set_cpu_math_library_num_threads(args.cpu_threads)

    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()

    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    # create predictor
    predictor = inference.create_predictor(config)

    # get input and output tensor property
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])

    return predictor, config, input_tensor, output_tensor


def preprocess(img_path, transforms):
    with open(img_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
    img = transforms(img)
    img = np.expand_dims(img, axis=0)
    return img


def postprocess(output):
    output = output.flatten()
    class_id = output.argmax()
    prob = output[class_id]
    return class_id, prob


def get_args(add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="PaddlePaddle Classification Training", add_help=add_help)

    parser.add_argument(
        "--model-dir", default=None, help="inference model dir")
    parser.add_argument(
        "--use-gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument(
        "--use-mkldnn", default=False, type=str2bool, help="use_mkldnn")
    parser.add_argument(
        "--min-subgraph-size", default=15, type=int, help="min_subgraph_size")
    parser.add_argument(
        "--max-batch-size", default=16, type=int, help="max_batch_size")
    parser.add_argument(
        "--cpu-threads", default=10, type=int, help="cpu-threads")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")

    parser.add_argument(
        "--resize-size", default=256, type=int, help="resize_size")
    parser.add_argument("--crop-size", default=224, type=int, help="crop_szie")
    parser.add_argument("--img-path", default="./images/demo.jpg")

    parser.add_argument(
        "--benchmark", default=False, type=str2bool, help="benchmark")
    parser.add_argument("--warmup", default=0, type=int, help="warmup iter")

    args = parser.parse_args()
    return args


def get_infer_gpuid():
    cmd = "env | grep CUDA_VISIBLE_DEVICES"
    env_cuda = os.popen(cmd).readlines()
    if len(env_cuda) == 0:
        return 0
    else:
        gpu_id = env_cuda[0].strip().split("=")[1]
        return int(gpu_id[0])


def predict(args):
    # init inference engine
    predictor, config, input_tensor, output_tensor = load_predictor(
        os.path.join(args.model_dir, "inference.pdmodel"),
        os.path.join(args.model_dir, "inference.pdiparams"), args)

    assert args.batch_size == 1, "batch size just supports 1 now."

    # init benchmark
    if args.benchmark:
        import auto_log
        pid = os.getpid()
        gpu_id = get_infer_gpuid()
        autolog = auto_log.AutoLogger(
            model_name="classification",
            model_precision=args.precision,
            batch_size=args.batch_size,
            data_shape="dynamic",
            save_path=None,
            inference_config=config,
            pids=pid,
            process_name=None,
            gpu_ids=gpu_id if args.use_gpu else None,
            time_keys=[
                "preprocess_time", "inference_time", "postprocess_time"
            ],
            warmup=0,
            logger=None)

    # build transforms
    infer_transforms = Compose([
        ResizeImage(args.resize_size), CenterCropImage(args.crop_size),
        NormalizeImage(), ToCHW()
    ])

    # wamrup
    if args.warmup > 0:
        for _ in range(args.warmup):
            x = paddle.rand([1, 3, args.crop_size, args.crop_size])
            input_tensor.copy_from_cpu(x)
            predictor.run()
            output = output_tensor.copy_to_cpu()

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # preprocess
    img = preprocess(args.img_path, infer_transforms)

    if args.benchmark:
        autolog.times.stamp()

    # inference using inference engine
    input_tensor.copy_from_cpu(img)
    predictor.run()
    output = output_tensor.copy_to_cpu()

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    class_id, prob = postprocess(output)

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    print(f"image_name: {args.img_path}, class_id: {class_id}, prob: {prob}")
    return class_id, prob


if __name__ == "__main__":
    args = get_args()
    class_id, prob = predict(args)

    reprod_logger = ReprodLogger()
    reprod_logger.add("class_id", np.array([class_id]))
    reprod_logger.add("prob", np.array([prob]))
    reprod_logger.save("output_inference_engine.npy")