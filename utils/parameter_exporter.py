import os
import yaml

os.makedirs("../parameters", exist_ok=True)

model_parameters = {}
input_channel = [3]
conv_channel = [8]
fc1_dim = [256]
fc2_dim = [128]

learning_parameters = {}
epochs = [50]
batch_size = [32]
lr = [0.0001]
seed = [0]

param = { 
            "model_parameters": model_parameters,
            "learning_parameters": learning_parameters}

cnt = 0
for i in input_channel:
    param["model_parameters"].update(input_channel=i)
    for i in conv_channel:
        param["model_parameters"].update(conv_channel=i)
        for i in conv_channel:
            param["model_parameters"].update(conv_channel=i)
            for i in lr:
                param["learning_parameters"].update(lr=i)
                for i in epochs:
                    param["learning_parameters"].update(epochs=i)
                    for i in batch_size:
                        param["learning_parameters"].update(batch_size=i)
                        for i in seed:
                            param["learning_parameters"].update(seed=i)
                            for i in fc1_dim:
                                param["model_parameters"].update(fc1_dim=i)
                                for i in fc2_dim:
                                    param["model_parameters"].update(fc2_dim=i)
                                    file_name = "param" + str(cnt) + ".yaml"
                                    with open(os.path.join("../parameters", file_name), "w") as f:
                                        yaml.dump(param, f)
                                    print("Successfully saved parameters to", file_name)
                                    cnt += 1
