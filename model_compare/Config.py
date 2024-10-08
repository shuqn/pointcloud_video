import Hyperparameters
class Config:
    ######QOE参数######
    N = 4###窗口大小5gpf
    tile_num = Hyperparameters.TILE_IN_F  # 切块数量  ###每个F切为2*2*2个tile
    level_num = Hyperparameters.QUALITY_LEVELS  # 质量等级数量
    group_of_x = N * tile_num  # 决策变量x的组数（5个一组）###=40

    x_num = level_num * group_of_x ###160
    QSP_num = x_num ###160
    fov_num = group_of_x ###40
    fov_1_num = tile_num###能不能看到每个tile 8
    x_1_num = tile_num * level_num###32
    bf0_num = 1
    Dkc_num = group_of_x###40
    BW_num = N###5

    #c_num = 3*QSP_num + fov_num + fov_1_num + x_1_num + bf0_num + Dkc_num
    c_num = 2 * QSP_num + fov_num + fov_1_num + x_1_num + bf0_num + Dkc_num + BW_num###446
    s_len = x_num + c_num###606
    #######DQN参数#######
    num_episodes = 200000  # 训练的总episode数量
    num_exploration_episodes = 120000  # 探索过程所占的episode数量
    # max_len_episode = 1500          # 每个episode的最大回合数
    memory_size = 200000    #池大小
    batch_size = 1  # 批次大小
    learning_rate = (1e-1)*1.5# 学习率
    gamma = 1.  # 折扣因子
    initial_epsilon = 1.  # 探索起始时的探索率
    final_epsilon = 0.01  # 探索终止时的探索率
