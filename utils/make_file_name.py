
def make_file_name(params=None): #パラメータからファイル名をつくるプログラム
    return 'input_ch-' + str(params.input_channel) + '_conv_channel-' + str(params.conv_channel) + \
        '_fc1_dim-' + str(params.fc1_dim) + '_fc2_dim-' + str(params.fc2_dim) + \
        '_epochs-' + str(params.epochs) + '_lr-' + str(params.lr) + \
        '_BatchSize-' + str(params.batch_size) + '_seed-' + str(params.seed)
