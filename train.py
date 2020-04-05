import paddle.fluid as fluid
from model import MSCNN
from load import get_reader


train_reader, test_reader = get_reader(16)
image = fluid.layers.data(name='image', shape=(3, 224, 224), dtype='float32')
label = fluid.layers.data(name='label', shape=(1, 56, 56), dtype='float32')
out = MSCNN(image)


cost = fluid.layers.square_error_cost(input=out, label=label)
avg_cost = fluid.layers.mean(x=cost)

test_program = fluid.default_main_program().clone(for_test=True)

optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
optimizer = optimizer.minimize(avg_cost)

# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace() 
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())


train_costs = []
test_costs = []

test_cost_min = 1000000

feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

for i in range(200):
    for batch_id, data in enumerate(train_reader()):
        cost = exe.run(program=fluid.default_main_program(), 
                            fetch_list=[avg_cost], 
                            feed=feeder.feed(data))
        if batch_id % 4 == 0:
            print('第 ', i+1, ' 次迭代，第 ', batch_id+1, ' 批次：', end=' ')
            print('cost=%.4f' % cost[0])
            train_costs.append(cost[0])
    
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
        fluid.io.save_inference_model(dirname='./model', feeded_var_names=['image'], target_vars=[out], executor=exe)
    print('Test：%d, Cost：%.4f' % (i, test_cost))

print('模型保存完毕')