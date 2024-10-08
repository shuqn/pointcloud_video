# import scddqn
# state=[0.07585353846153846, 0.1327436923076923, 0.18963384615384615, 0.246524, 0.07611323076923077, 0.13319815384615386, 0.19028307692307692, 0.24736799999999998, 0.07242953846153846, 0.12675169230769232, 0.18107384615384614, 0.235396, 0.07733538461538461, 0.13533692307692308, 0.19333846153846151, 0.25134, 0.00023507692307692308, 0.0004113846153846154, 0.0005876923076923077, 0.000764, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.128864, 0.22551200000000002, 0.32216, 0.418808, 0.1274523076923077, 0.22304153846153846, 0.3186307692307693, 0.41422000000000003, 0.1341870769230769, 0.2348273846153846, 0.33546769230769224, 0.43610799999999994, 0.1355286153846154, 0.23717507692307693, 0.33882153846153845, 0.440468, 0.015016615384615385, 0.026279076923076926, 0.03754153846153846, 0.048804, 0.012572307692307692, 0.02200153846153846, 0.03143076923076923, 0.04086, 0.013902769230769229, 0.02432984615384615, 0.034756923076923074, 0.045183999999999995, 0.021084307692307694, 0.03689753846153847, 0.052710769230769235, 0.068524, 0.09695507692307692, 0.1696713846153846, 0.2423876923076923, 0.315104, 0.10067569230769233, 0.17618246153846157, 0.25168923076923083, 0.32719600000000004, 0.10680984615384616, 0.18691723076923078, 0.2670246153846154, 0.347132, 0.10364799999999999, 0.181384, 0.25911999999999996, 0.336856, 0.012971076923076922, 0.022699384615384614, 0.0324276923076923, 0.042156, 0.013976615384615384, 0.024459076923076924, 0.03494153846153846, 0.045424, 0.01757046153846154, 0.030748307692307693, 0.043926153846153844, 0.057104, 0.028251076923076927, 0.04943938461538462, 0.07062769230769232, 0.09181600000000001, 0.15341538461538462, 0.26847692307692306, 0.38353846153846155, 0.49860000000000004, 0.17021784615384614, 0.29788123076923073, 0.42554461538461535, 0.5532079999999999, 0.1803150769230769, 0.31555138461538457, 0.4507876923076923, 0.586024, 0.19281107692307692, 0.3374193846153846, 0.48202769230769227, 0.626636, 0.06268800000000001, 0.10970400000000001, 0.15672000000000003, 0.20373600000000003, 0.04732430769230769, 0.08281753846153847, 0.11831076923076923, 0.153804, 0.03594707692307692, 0.06290738461538462, 0.0898676923076923, 0.116828, 0.028345846153846152, 0.04960523076923076, 0.07086461538461539, 0.092124, 0.16013907692307694, 0.28024338461538467, 0.40034769230769235, 0.520452, 0.1531212307692308, 0.26796215384615385, 0.38280307692307697, 0.4976440000000001, 0.15207507692307692, 0.2661313846153846, 0.38018769230769234, 0.494244, 0.14121846153846154, 0.2471323076923077, 0.35304615384615384, 0.45896000000000003, 0.11521969230769231, 0.20163446153846154, 0.2880492307692308, 0.374464, 0.11776246153846154, 0.20608430769230768, 0.2944061538461539, 0.382728, 0.11721723076923077, 0.20513015384615385, 0.2930430769230769, 0.380956, 0.11831876923076923, 0.20705784615384615, 0.29579692307692307, 0.384536, 0.09159261538461538, 0.16028707692307692, 0.22898153846153846, 0.297676, 0.08835323076923077, 0.15461815384615385, 0.2208830769230769, 0.287148, 0.08404307692307691, 0.1470753846153846, 0.21010769230769227, 0.27314, 0.07660307692307693, 0.13405538461538463, 0.19150769230769232, 0.24896000000000001, 0.08178584615384617, 0.14312523076923078, 0.2044646153846154, 0.26580400000000004, 0.08533907692307693, 0.14934338461538463, 0.21334769230769235, 0.27735200000000004, 0.08953846153846154, 0.1566923076923077, 0.22384615384615386, 0.291, 0.08863015384615386, 0.15510276923076927, 0.22157538461538465, 0.288048, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 29.373, 29.373, 56.852, 94.113, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1.0, 1.7147062891200495, 1.717517471074472, 1.7278626762164346, 1.7210427728311954, 1.9180949147390622, 1.9163980511277006, 1.917879943776935, 1.904951182900625, 1.3540765746248806, 1.3562765232230667, 1.3609741309304229, 1.3569432633365663, 1.603868390469695, 1.6006877379320583, 1.5953070663486135, 1.5837126643537616, 1.1944916081529624, 1.19485051426781, 1.194190128188062, 1.1912866212270203, 1.4716251716887183, 1.4664184185852445, 1.4556249905596634, 1.4443041913788417, 1.6673027699154854, 1.6723077878626438, 1.685703165513377, 1.6822297913526827, 1.8758382047948192, 1.8759873571711643, 1.8799864927977061, 1.8699593188715335, 1.2935239613427538, 1.2985503768995834, 1.3070334040956335, 1.3073652377642946, 1.5530855361401865, 1.5520787055642666, 1.5495452140230759, 1.54144563583491, 1.1253848544244098, 1.1288998296332806, 1.132332016955195, 1.1344922471813061, 1.4161078753286596, 1.4131985183497928, 1.4053219728951154, 1.3978281161267612]
# bit_rate=scddqn.get_bitrate(state)
###在import其他包之前，开头必须加上这三行，没有任何作用，但是不加会报错
import buffer_based_back
import buffer_based
import random_choose_back
import random_choose
import load_data
import random
import os
import matplotlib.pyplot as plt
import Hyperparameters
# import offline
import pot_drl
import pot_drl_back
import bw_cons
import bw_cons_back
import bw_cons_all
RANDOM_SEED=Hyperparameters.RANDOM_SEED
random.seed(RANDOM_SEED)
TEST_TIME=40

# RESULT=['bb','rc']
RESULT=['bb_back','bb','random','random_back','pot_drl','pot_drl_back','bw_cons','bw_cons_back','bw_cons_all']
TEST_MODEL=[bw_cons]
TEST_MODEL=[bw_cons]
# TEST_MODEL=[buffer_based_back,buffer_based,random_choose,random_choose_back,pot_drl,pot_drl_back,bw_cons,bw_cons_back,bw_cons_all]
# RESULT=['bb']
# TEST_MODEL=[bw_cons,bw_cons_back,bw_cons_all]
#运行前清空result
current_directory = str(os.path.dirname(os.path.realpath(__file__)))
folder_path=current_directory+'/results'
# if os.path.exists(folder_path):
#     # 删除文件夹下的所有文件
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         if os.path.isfile(file_path):
#             os.remove(file_path)
trace=[0,714, 486, 1196, 352, 1312, 1192, 1062, 97, 82, 847, 1198, 94, 892, 1040, 1349, 1204, 1054, 300, 614, 387, 386, 807, 1297, 909, 1453, 128, 683, 632, 271, 1265, 1226, 1364, 1420, 597, 1288, 132, 245, 744, 209, 1285]
for time in range(1,TEST_TIME+1):
    print(time)
    bw_count=30000
    init_bw=random.uniform(0.2, 6)
    # frame_count=5000
    # min_base_random_size=random.uniform(0.1, 0.3)
    # max_base_random_size=random.uniform(1.3, 1.7)*min_base_random_size
    # cooked_time,cooked_bw = load_data.load_trace(filename=current_directory+'/cooked_data/trace_5g.txt',startposition=random.randint(0,1476))
    cooked_time,cooked_bw = load_data.load_trace(filename=current_directory+'/cooked_data/trace_5g.txt',startposition=trace[time])
    # weights = [random.random()]
    # weights.append(1-weights[0])
    # video_size=load_data.load_video(frame_count=frame_count,min_base_random_size=min_base_random_size,max_base_random_size=max_base_random_size)
    video_size=load_data.load_video(filename=current_directory+'/cooked_data/tile_counts_1.txt')
    fov=load_data.load_fov(filename=current_directory+'/cooked_data/p'+str(time)+'_fovs_1.txt')
    dis=load_data.load_dis(filename=current_directory+'/cooked_data/p'+str(time)+'_dis_1.txt')
    
    for model in TEST_MODEL:
        model.test(cooked_time,cooked_bw,video_size,fov,dis,time)
    if time==38:
        print(cooked_bw)
# for time in range(1,TEST_TIME+1):
#     print(time)
#     bw_count=30000
#     init_bw=random.uniform(0.2, 6)
#     # frame_count=5000
#     # min_base_random_size=random.uniform(0.1, 0.3)
#     # max_base_random_size=random.uniform(1.3, 1.7)*min_base_random_size
#     # cooked_time,cooked_bw = load_data.load_trace(filename=current_directory+'/cooked_data/trace_5g.txt',startposition=random.randint(0,1476))
#     cooked_time,cooked_bw = load_data.load_trace(filename=current_directory+'/cooked_data/trace_5g.txt',startposition=trace2[time])
#     # weights = [random.random()]
#     # weights.append(1-weights[0])
#     # video_size=load_data.load_video(frame_count=frame_count,min_base_random_size=min_base_random_size,max_base_random_size=max_base_random_size)
#     video_size=load_data.load_video(filename=current_directory+'/cooked_data/tile_counts_1.txt')
#     fov=load_data.load_fov(filename=current_directory+'/cooked_data/p'+str(time)+'_fovs_1.txt')
#     dis=load_data.load_dis(filename=current_directory+'/cooked_data/p'+str(time)+'_dis_1.txt')
    
#     for model in TEST_MODEL:
#         model.test(cooked_time,cooked_bw,video_size,fov,dis,time)
#     if time==38:
#         print(cooked_bw)
# output=[]
# for r in range(len(RESULT)):
#     output.append([])
#     with open(folder_path+'/result_sim_'+RESULT[r], 'r') as file1:
#         for line in file1:
#             output[r].append(float(line.strip()))
# # 绘制折线图
#     plt.plot(output[r], label=RESULT[r])
# plt.legend()
# plt.xticks(range(len(output[0])))
# plt.xlabel('video_index')
# plt.ylabel('QoE')


# # 显示图形
# plt.show()