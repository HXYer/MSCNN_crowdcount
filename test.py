import paddle.fluid as fluid
import cv2


# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace() 
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())
[infer_program, feeded_var_names, target_vars] = fluid.io.load_inference_model(dirname='./model4', executor=exe)

def load_image(path):
    img = paddle.dataset.image.load_image(file=path, is_color=True)
    img = paddle.dataset.image.simple_transform(im=img, resize_size=224, crop_size=224, is_color=True, is_train=False)
    img = img.astype('float32')
    img = img[np.newaxis, ] / 255.0 * 2 - 1
    return img
	
with open('stage1/real.json') as f:
    data = json.load(f)

cost = 0
good = 0
error_rate = 0
n = len(data)
for path, num in data.items():
    img = load_image(path)
    result = exe.run(program=infer_program, 
                feed={feeded_var_names[0]:img}, 
                fetch_list=target_vars)
    label = np.rint(np.sum(result) / UNIT)
    if label >= 20:
        label = 19
    cost += abs(label - num)
    error_rate += abs(label - num) / num
    if label == num:
        good += 1

cost /= n
good /= n
error_rate /= n
print("绝对值损失值为：", cost)
print("平均错误率为：", error_rate)
print("预测值与真实值相等的比例为：", good)
