import paddle.fluid as fluid
from model import Model
from proprecessing import Proprecessing
from load import Data_reader

def cost_visualization(train_costs, test_costs):
    p1 = plt.figure(figsize=(12,10))
    ax1 = p1.add_subplot(2,1,1)
    plt.plot(train_costs, label='train cost')
    plt.title('cost')
    plt.xlabel('item')
    plt.ylabel('cost')
    plt.grid()
    plt.legend()

    ax2 = p1.add_subplot(2,1,2)
    plt.plot(test_costs, label='test cost', color='red')
    plt.title('cost')
    plt.xlabel('item')
    plt.ylabel('cost')
    plt.grid()
    plt.legend()

INPUT_SIZE = 224
BATCH_SIZE = 8
UNIT = 10
ker = 9
prep = Proprecessing(UNIT, ker)
train_outputs = prep.read_train_json()
train_data = prep.get_data(train_outputs, INPUT_SIZE)

data_read = Data_reader(BATCH_SIZE, INPUT_SIZE)
data_read.dataset_split(train_data)
train_reader, test_reader = data_read.get_reader()

image, label, predict = Model().generator()

cost = fluid.layers.square_error_cost(input=predict, label=label)
avg_cost = fluid.layers.mean(cost)

optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-5)
optimizer.minimize(avg_cost)

test_program = fluid.default_main_program().clone(for_test=True)

place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace() 
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())

feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

train_costs = []
test_costs = []
test_cost_min = 1000000

for i in range(500):
    for batch_id, data in enumerate(train_reader()):
        train_cost = exe.run(program=fluid.default_main_program(), 
                            fetch_list=[avg_cost], 
                            feed=feeder.feed(data))
        if batch_id % 50 == 0:
            print('第 ', i+1, ' 次迭代，第 ', batch_id+1, ' 批次：', end=' ')
            print('cost= %.8f' % train_cost[0])
            train_costs.append(train_cost[0])
    
    test_this_costs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost = exe.run(program=test_program, 
                                    fetch_list=[avg_cost], 
                                    feed=feeder.feed(data))
        test_this_costs.append(test_cost[0])
    test_cost = sum(test_this_costs) / len(test_this_costs)
    test_costs.append(test_cost)
    if test_cost < test_cost_min:
        test_cost_min = test_cost
        if i < 100:
            fluid.io.save_inference_model(dirname='./model', feeded_var_names=['image'], target_vars=[predict], executor=exe)
        elif i < 200:
            fluid.io.save_inference_model(dirname='./model2', feeded_var_names=['image'], target_vars=[predict], executor=exe)
        elif i < 300:
            fluid.io.save_inference_model(dirname='./model3', feeded_var_names=['image'], target_vars=[predict], executor=exe)
        elif i < 400:
            fluid.io.save_inference_model(dirname='./model4', feeded_var_names=['image'], target_vars=[predict], executor=exe)
        else:
            fluid.io.save_inference_model(dirname='./model5', feeded_var_names=['image'], target_vars=[predict], executor=exe)
    print('Test：%d, Cost：%.8f' % (i+1, test_cost))

print('模型保存完毕')
cost_visualization(train_costs, test_costs)