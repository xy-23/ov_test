#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

const std::string keys =
  "{help h usage ? |     | 输出命令行参数说明}"
  "{@model-path    |     | 模型文件路径 }"
  "{device d       | CPU | 设备类型}";

void print_devices(const ov::Core & core)
{
  std::cout << "devices: ";
  for (const auto & device : core.get_available_devices()) std::cout << device << " ";
  std::cout << "\n";
}

void print_model_info(const ov::Model & model)
{
  std::cout << "model name: " << model.get_friendly_name() << "\n";
  std::cout << "  inputs: ";
  for (const auto & input : model.inputs())
    std::cout << input.get_element_type() << " " << input.get_shape() << " ";
  std::cout << "\n";
  std::cout << "  outputs: ";
  for (const auto & output : model.outputs())
    std::cout << output.get_element_type() << " " << output.get_shape() << " ";
  std::cout << "\n";
}

void print_compiled_model_info(const ov::CompiledModel & compiled_model)
{
  std::cout << "num_streams: " << compiled_model.get_property(ov::num_streams) << "\n";
  std::cout << "optimal_number_of_infer_requests: "
            << compiled_model.get_property(ov::optimal_number_of_infer_requests) << "\n";
}

class Model
{
public:
  Model(const std::string & model_path, const std::string & device, int height, int width)
  {
    ov::Core core;
    print_devices(core);

    auto model = core.read_model(model_path);
    print_model_info(*model);

    ov::preprocess::PrePostProcessor ppp(model);
    auto & input = ppp.input();

    input.tensor()
      .set_element_type(ov::element::u8)
      .set_shape({1, height, width, 3})
      .set_layout("NHWC")
      .set_color_format(ov::preprocess::ColorFormat::BGR);

    input.model().set_layout("NCHW");

    input.preprocess()
      .convert_element_type(ov::element::f32)
      .convert_color(ov::preprocess::ColorFormat::RGB)
      .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
      .scale(255);

    model = ppp.build();

    compiled_model_ = core.compile_model(
      model, device, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));

    print_compiled_model_info(compiled_model_);
  }

  void infer(cv::Mat img)
  {
    auto input_port = compiled_model_.input();
    auto infer_request = compiled_model_.create_infer_request();
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), img.data);
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
  }

private:
  ov::CompiledModel compiled_model_;
};

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help") || !cli.has("@model-path")) {
    cli.printMessage();
    return 0;
  }

  auto model_path = cli.get<std::string>("@model-path");
  std::cout << "model_path: " << model_path << "\n";

  auto device = cli.get<std::string>("device");
  std::cout << "device: " << device << "\n";

  Model model(model_path, device, 1024, 1280);
  cv::VideoCapture cap("video.avi");

  cv::Mat img;
  int count = 0;
  double total_time = 0;

  while (true) {
    auto success = cap.read(img);
    if (!success) break;

    auto start = std::chrono::steady_clock::now();
    model.infer(img);
    auto end = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration<double>(end - start).count();
    // std::cout << duration * 1e3 << " ms\n";

    count++;
    total_time += duration;
  }

  std::cout << "average: " << count / total_time << " fps\n";

  return 0;
}