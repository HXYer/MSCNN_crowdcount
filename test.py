import paddle.fluid as fluid
import cv2


# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace() 
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())
[infer_program, feeded_var_names, target_vars] = fluid.io.load_inference_model(dirname='./model', executor=exe)

def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, [2, 0, 1])
    # img = paddle.dataset.image.load_image(file=img, is_color=True)
    # img = paddle.dataset.image.simple_transform(im=img, resize_size=224, crop_size=224, is_color=True, is_train=True)
    img = img.astype('float32')
    img = img[np.newaxis, ] / 255.0 * 2 - 1
    return img
	
with open('stage1/real.json') as f:
    data = json.load(f)
    n = len(data)
    cost = 0
    good = 0
    bias = 0
    for path, value in data.items():
        img = load_image(path)
        result = exe.run(program=infer_program, 
                    feed={feeded_var_names[0]:img}, 
                    fetch_list=target_vars)
        res = int(np.sum(result))
        label = int(np.sum(value))
        print(res, '\t', label)
        cost += (res-label)**2
        bias += abs(res - label)
        if label == res:
            good += 1

    cost /= n
    bias /= n
    good /= n
    print('平方差损失值为：', cost)
    print("预测值与真实值平均差：", bias)
    print("预测值与真实值相等的比例：", good)
