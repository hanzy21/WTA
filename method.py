import numpy as np
import math
from rrt import RRTPlanner
import geatpy as ea
from matplotlib import pyplot as plt
from sko.PSO import PSO

#双方单位类的初始化，属性包括位置、速度、威胁度（<1的百分数）、重要程度（在1上下）
class Agent():

    def __init__(self, pos, speed, threat, value):
        self.pos = pos
        self.speed = speed
        self.threat = threat
        self.value = value
        
#使用geatpy求解的问题的初始化
class problem(ea.Problem):
    
    def __init__(self, matrix_a, matrix_e):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = matrix_a.shape[0]+matrix_a.shape[1]+2  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = np.hstack([np.zeros((Dim-2)),0,0])  # 决策变量下界
        ub = np.hstack([np.ones((Dim-2)),2,2]) # 决策变量上界
        lbin = np.ones((Dim)) # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = np.ones((Dim))  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        self.matrix_a = matrix_a
        self.matrix_e = matrix_e
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        size_a = self.matrix_a.shape[0]
        size_e = self.matrix_a.shape[1]
        ObjV = np.zeros((Vars.shape[0],))
        CV1 = np.zeros((Vars.shape[0],size_e))
        CV2 = np.zeros((Vars.shape[0],size_a))
        CV3 = -1*np.ones((Vars.shape[0],))
        CV4 = -1*np.ones((Vars.shape[0],))
        CV5 = 0.9*np.ones((Vars.shape[0],))
        CV6 = 0.9*np.ones((Vars.shape[0],))
        for i in range(Vars.shape[1]):
            if i < size_a:
                #CV2 += np.transpose(np.tile(Vars[:,i],[size_a,1]))*np.tile(self.matrix_a[:,i],[Vars.shape[0],1])
                CV1 += (np.transpose(np.tile(Vars[:,i],[size_e,1]))
                        *np.tile(self.matrix_e[i,:],[Vars.shape[0],1]))
                CV3 += Vars[:,i]
                CV5 -= Vars[:,i]
                #for j in range(int(Vars.shape[1]/2-1)):
                ObjV += (np.transpose(np.tile(Vars[:,i],[size_e,1]))
                         *np.tile(self.matrix_a[i,:]+self.matrix_e[i,:],[Vars.shape[0],1])
                         *Vars[:,size_a:size_e+size_a]).sum(axis=1)
            if i >= size_a and i < size_a + size_e:
                CV2 += (np.transpose(np.tile(Vars[:,i],[size_a,1]))
                        *np.tile(self.matrix_a[:,i-size_a],[Vars.shape[0],1]))
                #CV1 += np.transpose(np.tile(Vars[:,i],[size_e,1]))*np.tile(self.matrix_e[i-size_a,:],[Vars.shape[0],1])
                CV4 += Vars[:,i]
                CV6 -= Vars[:,i]
            if i == Vars.shape[1]-2:
                CV2 -= np.transpose(np.tile(Vars[:,i],[size_a,1]))
                ObjV -= Vars[:,i]
            if i == Vars.shape[1]-1:
                CV1 -= np.transpose(np.tile(Vars[:,i],[size_e,1]))
                ObjV -= Vars[:,i]
        
        ObjV = np.resize(ObjV,(ObjV.shape[0],1))
        pop.ObjV = ObjV  # 计算目标函数值，赋值给pop种群对象的ObjV属性
        # 采用可行性法则处理约束
        CV3 = np.resize(CV3,(CV3.shape[0],1))
        CV4 = np.resize(CV4,(CV4.shape[0],1))
        CV5 = np.resize(CV5,(CV5.shape[0],1))
        CV6 = np.resize(CV6,(CV6.shape[0],1))
        pop.CV = np.hstack([CV1,CV2,CV3,CV4,CV5,CV6])

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值）
        referenceObjV = np.array([[0]])
        return referenceObjV
    
def func(matrix_a, matrix_e,x):
    size_a = matrix_a.shape[0]
    size_e = matrix_a.shape[1]
    ObjV = 0
    for i in range(len(x)):
            if i < size_a:
                ObjV += x[i]*(matrix_a[i,:]+matrix_e[i,:])*x[size_a:size_e+size_a].sum(axis=1)
                
            if i >= len(x)-2:
                ObjV -= x[i]
                
'''
def runpso(func,matrix_a, matrix_e):
    dim = matrix_a.shape[0]+matrix_a.shape[1]+2
    pop = 400 #种群数量
    max_iter = 10000 #最大迭代次数
    lb = np.hstack([np.zeros((dim-2)),0,0])  # 决策变量下界
    ub = np.hstack([np.ones((dim-2)),2,2]) # 决策变量上界
    constraint_ueq = (
    lambda x: (x[0] - 1) ** 2 + (x[1] - 0) ** 2 - 0.5 ** 2)
    pso = PSO(func=demo_func, n_dim=dim, pop=pop, max_iter=max_iter, 
              lb=lb, ub=ub, constraint_ueq=constraint_ueq)
        
'''
#生成策略集
def strategy(agent_list_a, agent_list_e):
    a = len(agent_list_a)
    e = len(agent_list_e)
    stra_a = np.zeros((pow(e+1,a),a))#我方策略集
    stra_e = np.zeros((pow(a+1,e),e))#敌方策略集
    #使用枚举法列出双方可能的所有策略
    for i in range(pow(e+1,a)):
        s = 1
        k = i
        for j in range(a):
            if s != 0:                
                s = k//(e+1)  # 商
                y = k%(e+1)  # 余数
                stra_a[i][j] = y
                k = s
            else:
                stra_a[i][j] = 0
    for i in range(pow(a+1,e)):
        s = 1
        k = i
        for j in range(e):
            if s != 0:                
                s = k//(a+1)  # 商
                y = k%(a+1)  # 余数
                stra_e[i][j] = y
                k = s
            else:
                stra_e[i][j] = 0                
    return stra_a, stra_e
                
#计算各个单位之间所需时间
def compute_timetable(agent_list_a, agent_list_e, field, threshold, obstacle):
    a = len(agent_list_a)
    e = len(agent_list_e)    
    timetable_a = np.zeros((a,e))
    timetable_e = np.zeros((a,e))
    pathtable = []
    impassable_area = np.zeros(field.shape)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            if field[i][j] > threshold or obstacle[i][j] == 1:
                impassable_area[i][j] = 1

    for i in range(a):
        for j in range(e):
            rrt = RRTPlanner(agent_list_a[i].pos,agent_list_e[j].pos,impassable_area)
            dist,path = rrt.RRT()
            timetable_a[i][j] = dist/agent_list_a[i].speed
            timetable_e[i][j] = dist/agent_list_e[j].speed
            pathtable.append(path)
    return timetable_a, timetable_e, pathtable, impassable_area

#计算策略集对应的双方胜率双矩阵
def win_matrix(agent_list_a, agent_list_e, stra_a, stra_e):
    a = len(agent_list_a)
    e = len(agent_list_e)
    m = stra_a.shape[0]
    n = stra_e.shape[0]
    #矩阵和列表的初始化
    win_matrix_a = np.zeros((m,n))
    win_matrix_e = np.zeros((m,n))
    win_list_a = np.zeros(m)
    win_list_e = np.zeros(n)  
    #胜率用Sigma(敌方毁伤概率*重要程度-我方毁伤概率*重要程度)表示
    #计算我方策略对应的我方胜率
    for i in range(m):
        for j in range(1,e+1):
            temp_tuple = np.where(stra_a[i] == j) #我方策略i中打击敌方单位j的单位列表                
            prob = 1 #无法毁伤的概率
            for k in range(len(temp_tuple[0])):
                prob = prob*(1 - agent_list_a[temp_tuple[0][k]].threat)                   
            win_list_a[i] += (1-prob)*agent_list_e[j-1].value
    #敌方策略对应的敌方胜率
    for i in range(n):
        for j in range(1,a+1):
            temp_tuple = np.where(stra_e[i] == j) #敌方策略i中打击我方单位j的单位列表                
            prob = 1 #无法毁伤的概率
            for k in range(len(temp_tuple[0])):
                prob = prob*(1 - agent_list_e[temp_tuple[0][k]].threat)
            win_list_e[i] += (1-prob)*agent_list_a[j-1].value
    #双方胜率叠加得到综合胜率
    #print(win_list_a,win_list_e)
    for i in range(m):
        for j in range(n):
            win_matrix_a[i][j] = win_list_a[i] - win_list_e[j]
            win_matrix_e[i][j] = -1*win_matrix_a[i][j]
    
    #胜率矩阵要提前归一化
    max_win_a = np.max(win_matrix_a)
    min_win_a = np.min(win_matrix_a)
    win_matrix_a = (win_matrix_a-min_win_a)/(max_win_a-min_win_a)
    win_matrix_a[np.where(win_matrix_a==0)] = 0.00001
    max_win_e = np.max(win_matrix_e)
    min_win_e = np.min(win_matrix_e)
    win_matrix_e = (win_matrix_e-min_win_e)/(max_win_e-min_win_e)  
    win_matrix_e[np.where(win_matrix_e==0)] = 0.00001
 
    return win_matrix_a, win_matrix_e

#计算双方每个单位到敌方每个单位所需时间双矩阵
def time_matrix(agent_list_a, agent_list_e, stra_a, stra_e, timetable_a, timetable_e, action_time):
    a = len(agent_list_a)
    e = len(agent_list_e)
    m = stra_a.shape[0]
    n = stra_e.shape[0]
    time_matrix_a = np.zeros((m,n))
    time_matrix_e = np.zeros((m,n))
    time_list_a = np.zeros(m)
    time_list_e = np.zeros(n)
    num_list_a = np.zeros(m)
    num_list_e = np.zeros(n)
    
    for i in range(m):
        temp = 0
        num_list_a[i] = np.where(stra_a[i]!=0)[0].shape[0]
        for j in range(a):#查看我方每个单位在该策略中的耗时          
            if stra_a[i][j] != 0:    
                time = timetable_a[j][int(stra_a[i][j]-1)]
                if time > temp:
                    temp = time
        if temp == 0:
            temp = 200
        if i == 0:
            print(temp)
        time_list_a[i] = temp
        
    for i in range(n):
        temp = 0
        num_list_e[i] = np.where(stra_e[i]!=0)[0].shape[0]
        for j in range(e):#查看敌方每个单位在该策略中的耗时          
            if stra_e[i][j] != 0:    
                time = timetable_e[int(stra_e[i][j]-1)][j]
                if time > temp:
                    temp = time
        if temp == 0:
            temp = 200
        time_list_e[i] = temp
    '''
    #对不进行任何行动的决策的惩罚
    for i in range(m):
        if time_list_a[i] == 0:
            time_list_a[i] = max(time_list_a)*2
    for i in range(n):
        if time_list_e[i] == 0:
            time_list_e[i] = max(time_list_e)*2
    '''
    #给出综合时间矩阵
    for i in range(m):
        for j in range(n):
            a = max(num_list_a[i],num_list_e[j])
            b = min(num_list_a[i],num_list_e[j])
            if a != 0:   
                time_matrix_a[i][j] = time_list_a[i] + action_time*((b+1)/a)
                time_matrix_e[i][j] = time_list_e[j] + action_time*((b+1)/a)
            else:
                time_matrix_a[i][j] = time_list_a[i]
                time_matrix_e[i][j] = time_list_e[j]
    return time_matrix_a, time_matrix_e

#资源消耗双矩阵  【问题】：资源是否要考虑重要程度？先考虑了
def resource_matrix(agent_list_a, agent_list_e, stra_a, stra_e):
    a = len(agent_list_a)
    e = len(agent_list_e)
    m = stra_a.shape[0]
    n = stra_e.shape[0]
    resource_matrix_a = np.zeros((m,n))
    resource_matrix_e = np.zeros((m,n))
    resource_list_a = np.zeros(m)
    resource_list_e = np.zeros(n)
    for i in range(m):
        for j in range(a):
            if stra_a[i][j] != 0:    
                resource_list_a[i] += agent_list_a[j].value
    for i in range(n):
        for j in range(e):
            if stra_e[i][j] != 0:    
                resource_list_e[i] += agent_list_e[j].value
    #给出综合资源消耗矩阵
    for i in range(m):
        for j in range(n):
            resource_matrix_a[i][j] = resource_list_a[i]
            resource_matrix_e[i][j] = resource_list_e[j]
    return resource_matrix_a, resource_matrix_e

#三个矩阵的化简
def matrix_simplify(win_matrix_a, win_matrix_e, time_matrix_a, time_matrix_e, 
                    resource_matrix_a, resource_matrix_e):
    
    #归一化  

    max_time_a = np.max(time_matrix_a)
    min_time_a = np.min(time_matrix_a)
    time_matrix_a = (time_matrix_a-min_time_a)/(max_time_a-min_time_a)   
    time_matrix_a[np.where(time_matrix_a==0)] = 0.00001
    max_time_e = np.max(time_matrix_e)
    min_time_e = np.min(time_matrix_e)
    time_matrix_e = (time_matrix_e-min_time_e)/(max_time_e-min_time_e) 
    time_matrix_e[np.where(time_matrix_e==0)] = 0.00001    
    max_resource_a = np.max(resource_matrix_a)
    min_resource_a = np.min(resource_matrix_a)
    resource_matrix_a = (resource_matrix_a-min_resource_a)/(max_resource_a-min_resource_a)    
    resource_matrix_a[np.where(resource_matrix_a==0)] = 0.00001
    max_resource_e = np.max(resource_matrix_e)
    min_resource_e = np.min(resource_matrix_e)
    resource_matrix_e = (resource_matrix_e-min_resource_e)/(max_resource_e-min_resource_e)       
    resource_matrix_e[np.where(resource_matrix_e==0)] = 0.00001
    
    
# =============================================================================
#     #熵权法合并
#     h_win = 0
#     h_time = 0
#     h_resource = 0
#     theta = 1/math.log(2*win_matrix_a.shape[0]*win_matrix_a.shape[1])
#     sum_win = np.sum(win_matrix_a)+np.sum(win_matrix_e)
#     sum_time = np.sum(time_matrix_a)+np.sum(time_matrix_e)
#     sum_resource = np.sum(resource_matrix_a)+np.sum(resource_matrix_e)
#     for i in range(win_matrix_a.shape[0]):
#         for j in range(win_matrix_a.shape[1]):
#             h_win -= (win_matrix_a[i][j]/sum_win)*math.log(win_matrix_a[i][j]/sum_win)
#             h_win -= (win_matrix_e[i][j]/sum_win)*math.log(win_matrix_e[i][j]/sum_win)
#             h_time -= (time_matrix_a[i][j]/sum_time)*math.log(time_matrix_a[i][j]/sum_time)
#             h_time -= (time_matrix_e[i][j]/sum_time)*math.log(time_matrix_e[i][j]/sum_time)  
#             h_resource -= (resource_matrix_a[i][j]/sum_resource)*math.log(resource_matrix_a[i][j]/sum_resource)
#             h_resource -= (resource_matrix_e[i][j]/sum_resource)*math.log(resource_matrix_e[i][j]/sum_resource)  
# 
#     omega_sum = 1-theta*h_win + 1-theta*h_time + 1-theta*h_resource
#     omega_win = (1-theta*h_win)/omega_sum
#     omega_time = (1-theta*h_time)/omega_sum
#     omega_resource = (1-theta*h_resource)/omega_sum
#     #得出单目标综合矩阵
# =============================================================================
    #胜率优先
    omega_win = 0.9
    omega_time = 0.05
    omega_resource = 0.05
    matrix_a1 = win_matrix_a*omega_win - time_matrix_a*omega_time - resource_matrix_a*omega_resource
    matrix_e1 = win_matrix_e*omega_win - time_matrix_e*omega_time - resource_matrix_e*omega_resource       
    matrix_a1 -= np.min(matrix_a1)
    matrix_e1 -= np.min(matrix_e1)
    
    #时间优先
    omega_win = 0.6
    omega_time = 0.35
    omega_resource = 0.05
    matrix_a2 = win_matrix_a*omega_win - time_matrix_a*omega_time - resource_matrix_a*omega_resource
    matrix_e2 = matrix_e1 #win_matrix_e*omega_win - time_matrix_e*omega_time - resource_matrix_e*omega_resource       
    matrix_a2 -= np.min(matrix_a2)
    matrix_e2 -= np.min(matrix_e2)
    
    #均衡策略
    omega_win = 0.75
    omega_time = 0.3
    omega_resource = 0.05
    matrix_a3 = win_matrix_a*omega_win - time_matrix_a*omega_time - resource_matrix_a*omega_resource
    matrix_e3 = matrix_e1 #win_matrix_e*omega_win - time_matrix_e*omega_time - resource_matrix_e*omega_resource       
    matrix_a3 -= np.min(matrix_a3)
    matrix_e3 -= np.min(matrix_e3)
    return matrix_a1, matrix_e1, matrix_a2, matrix_e2,  matrix_a3,  matrix_e3

#优超原则化简
def dominant(matrix_a, matrix_e):
    delete_a = np.array([])
    delete_e = np.array([])
    retain_a = np.array([])
    retain_e = np.array([])
    matrix_a_temp, retain_a = np.unique(matrix_a, return_index=True, axis = 0)
    matrix_e_temp, retain_e = np.unique(matrix_e, return_index=True, axis = 1)
    #print(matrix_a_temp,matrix_e_temp.shape)
    #print(retain_a)
    sum_a = matrix_a_temp.sum(axis=1)
    sum_e = matrix_e_temp.sum(axis=0)
    sum_a_sorted = np.sort(sum_a)
    sum_e_sorted = np.sort(sum_e)
    #print(sum_a_sorted)
    i = 0
    #出现以下三种情况之一时停止循环：每行都与其他行进行了比较or超过50行进行了比较or优超到只剩一行
    while(i <= min(matrix_a_temp.shape[0],50) 
          and delete_a.shape[0] < matrix_a_temp.shape[0]-1): 
        l = np.where(sum_a==sum_a_sorted[matrix_a_temp.shape[0]-i-1])[0][0] #和最大的行
        #print(l)
        for j in range(matrix_a_temp.shape[0]-i):
            s = np.where(sum_a==sum_a_sorted[j])
            for k in s[0]:  #防止sum相同的行的影响
                if ((matrix_a_temp[l,:] >= matrix_a_temp[k,:]).all() #每个元素都要>=且不能全=
                    and (matrix_a_temp[l,:] != matrix_a_temp[k,:]).any()):
                    #print("1",k)
                    delete_a = np.append(delete_a, k)
        i += 1
    retain_a = np.delete(retain_a, np.unique(delete_a).astype(int))
    #print('a',retain_a,delete_a)
            

    i = 0
    while(i <= min(matrix_e_temp.shape[1],50) 
          and delete_e.shape[0] < matrix_e_temp.shape[1]-1):
        l = np.where(sum_e==sum_e_sorted[matrix_e_temp.shape[1]-i-1])[0][0]
        for j in range(matrix_e_temp.shape[1]-i):
            s = np.where(sum_e==sum_e_sorted[j])
            for k in s[0]:
                if ((matrix_e_temp[:,l] >= matrix_e_temp[:,k]).all() 
                    and (matrix_e_temp[:,l] != matrix_e_temp[:,k]).any()):
                    delete_e = np.append(delete_e, k)
        i += 1
    retain_e = np.delete(retain_e, np.unique(delete_e).astype(int))
    #print('b',retain_a,retain_e)
        
    '''
    #清除重复项
    a_max = np.where(sum_a==sum_a_sorted[-1])
    print(a_max)
    j = 0
    for i in a_max[0]:
        j += 1
        if j != 1:
            delete_a = np.append(delete_a, i)
                
    e_max = np.where(sum_e==sum_e_sorted[-1])
    j = 0
    for i in e_max[0]:
        j += 1
        if j != 1:
            delete_e = np.append(delete_e, i)
    
    for i in range(matrix_a.shape[0]):
        print(i)
        for j in range(i+1,matrix_a.shape[0]): 
            if (matrix_a[i,:] >= matrix_a[j,:]).all():
                delete_a = np.append(delete_a,j)
                matrix_a = np.delete(matrix_a, j, axis=0)
            elif (matrix_a[i,:] < matrix_a[j,:]).all():
                delete_a = np.append(delete_a,i)
                matrix_a = np.delete(matrix_a, i, axis=0)
                
                
    for i in range(matrix_e.shape[1]):
        print(i)
        for j in range(i+1, matrix_e.shape[1]):
            if (matrix_e[:,i] >= matrix_e[:,j]).all():
                delete_e = np.append(delete_e,j)
            elif (matrix_e[:,i] < matrix_e[:,j]).all():
                delete_e = np.append(delete_e,i)
    '''
    #delete_e = np.unique(delete_e).astype(int)
    #delete_a = np.unique(delete_a).astype(int)
    retain_a = np.sort(retain_a).astype(int)
    retain_e = np.sort(retain_e).astype(int)
    print(retain_a.shape,retain_e.shape)
    '''
    #获取保留下来的行列号
    for i in range(matrix_a.shape[0]):
        if i not in delete_a:
            retain_a = np.append(retain_a,i)
    for i in range(matrix_e.shape[1]):
        if i not in delete_e:
            retain_e = np.append(retain_e,i)  
            
    if not delete_a == []:
        matrix_a = np.delete(matrix_a, delete_a, axis=0)
        matrix_e = np.delete(matrix_e, delete_a, axis=0)
    if not delete_e == []:       
        matrix_a = np.delete(matrix_a, delete_e, axis=1)
        matrix_e = np.delete(matrix_e, delete_e, axis=1)
        '''
    matrix_a = matrix_a[retain_a,:]
    matrix_a = matrix_a[:,retain_e]
    matrix_e = matrix_e[retain_a,:]
    matrix_e = matrix_e[:,retain_e]
    matrix_a = np.resize(matrix_a,(retain_a.shape[0],retain_e.shape[0]))
    matrix_e = np.resize(matrix_e,(retain_a.shape[0],retain_e.shape[0]))
    #print(matrix_a)
    #print(matrix_e.shape)
    return matrix_a,matrix_e, retain_a, retain_e
    
#查找有无纯策略解
def solve_pure(matrix_a, matrix_e, retain_a, retain_e):
    for i in range(matrix_a.shape[0]):
        for j in range(matrix_a.shape[1]):
            if matrix_a[i][j] == np.max(matrix_a[:,j]) and matrix_e[i][j] == np.max(matrix_e[i,:]):
                print("纯策略解为:",int(retain_a[i]),int(retain_e[j]))
                return True
    return False

#用遗传算法求解
def solve_ea(problem, matrix_shape, retain_a, retain_e):
    Encoding = 'RI'  # 编码方式
    NIND = 500  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 20  # 最大进化代数
    myAlgorithm.mutOper.F = 0.5  # 差分进化中的参数F
    myAlgorithm.recOper.XOVR = 0.3 # 重组概率
    myAlgorithm.mutOper.FixType = 1
    myAlgorithm.mutOper.Parallel = True
    myAlgorithm.logTras = 500  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """===========================调用算法模板进行种群进化========================"""
    print("开始使用遗传算法")
    #生成先验种群
# =============================================================================
#     if max(matrix_shape[0],matrix_shape[1]) < NIND:  
#         m = max(matrix_shape[0],matrix_shape[1])
#         n = min(matrix_shape[0],matrix_shape[1])
#         if matrix_shape[0] > matrix_shape[1]:
#             chrom = np.hstack([np.identity(m),np.vstack([np.identity(n),np.zeros((m-n,n))]),
#                                2*np.ones((m,2))])
#         else:
#             chrom = np.hstack([np.vstack([np.identity(n),np.zeros((m-n,n))]),np.identity(m),
#                                2*np.ones((m,2))])
#         prepop = ea.Population(Encoding, Field, m, chrom)
#     else:
#         
# =============================================================================
        
    prenind = int(NIND/4)
    Chrom = np.vstack([np.tile(np.hstack([np.ones((matrix_shape[0]))/matrix_shape[0],
                               np.ones((matrix_shape[1]))/matrix_shape[1],2,2]),[prenind,1]),
                      np.tile(np.hstack([1,np.zeros((matrix_shape[0]-1)),1,
                               np.zeros((matrix_shape[1]-1)),2,2]),[prenind,1]),
                      np.tile(np.hstack([np.zeros((matrix_shape[0]-1)),1,
                               np.zeros((matrix_shape[1]-1)),1,2,2]),[prenind,1])])
    #print(chrom.shape)
    #Chrom = ea.ri2bs(Chrom, Field)
    prepop = ea.Population(Encoding, Field, prenind*3, Chrom)
    myAlgorithm.call_aimFunc(prepop)
# =============================================================================
    [BestIndi, population] = myAlgorithm.run(prepop)  # 执行算法模板，得到最优个体以及最后一代种群
    BestIndi.save()  # 把最优个体的信息保存到文件中
    """==================================输出结果=============================="""
    print('评价次数：%s' % myAlgorithm.evalsNum)
    print('时间已过 %s 秒' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('最优的目标函数值为：%s' % BestIndi.ObjV[0][0])
        print('最优的控制变量值为：')
        for i in range(BestIndi.Phen.shape[1]):
            print(BestIndi.Phen[0, i])
        print(matrix_shape)
        x = np.argmax(BestIndi.Phen[0,0:matrix_shape[0]])
        y = np.argmax(BestIndi.Phen[0,matrix_shape[0]:matrix_shape[0]+matrix_shape[1]])
        print("最优解序号与值分别为：")
        print(int(retain_a[x]),BestIndi.Phen[0,x])
        print(int(retain_e[y]),BestIndi.Phen[0,y+matrix_shape[0]])
        return int(retain_a[x]),int(retain_e[y])
    else:
        print('没找到可行解。')
        return 0,0

def DesRateEnough(Assign_a1, agent_list_a, agent_list_e):
    threshold = 0
    rate = 0
    for i in agent_list_e:
        threshold += i.value
    for j in range(1,len(agent_list_e) + 1):
        temp_tuple = np.where(Assign_a1 == j) #我方策略中打击敌方单位j的单位列表                
        prob = 1 #无法毁伤的概率
        for k in range(len(temp_tuple[0])):
            prob = prob*(1 - agent_list_a[temp_tuple[0][k]].threat)                   
        rate += (1-prob)*agent_list_e[j-1].value
    return rate  > threshold

def ExtraAssignment(Assign_a1, Assign_e1, agent_list_a, agent_list_e):
    agent_list_a_new = []
    agent_list_e_new = []
    agent_num_list_a_new = []
    agent_num_list_e_new = []
    
    for i in range(1, len(agent_list_a) + 1):
        temp_tuple = np.where(Assign_e1 == i) #敌方策略中打击我方单位i的单位列表                
        prob = 1 #无法毁伤的概率
        for k in range(len(temp_tuple[0])):
            prob = prob*(1 - agent_list_e[temp_tuple[0][k]].threat)  
        if prob > 0.5:
            agent_list_a_new.append(agent_list_a[i-1])
            agent_num_list_a_new.append(i)
            
    for j in range(1, len(agent_list_e) + 1):
        temp_tuple = np.where(Assign_a1 == j) #我方策略中打击敌方单位j的单位列表                
        prob = 1 #无法毁伤的概率
        for k in range(len(temp_tuple[0])):
            prob = prob*(1 - agent_list_a[temp_tuple[0][k]].threat)   
            if prob > 0.5:
                agent_list_e_new.append(agent_list_e[j-1])
                agent_num_list_e_new.append(j)
                
    if agent_list_a_new and agent_list_e_new:
        stra_a_new, stra_e_new = strategy(agent_list_a_new, agent_list_e_new)
    else:
        print("aaaaa")
        return False
    m = stra_a_new.shape[0]
    win_list_a = np.zeros(m) 
    #胜率用Sigma(敌方毁伤概率*重要程度-我方毁伤概率*重要程度)表示
    #计算我方策略对应的我方胜率
    for i in range(stra_a_new.shape[0]):
        for j in range(1, len(agent_list_e_new) + 1):
            temp_tuple = np.where(stra_a_new[i] == j) #我方策略i中打击敌方单位j的单位列表                
            prob = 1 #无法毁伤的概率
            for k in range(len(temp_tuple[0])):
                prob = prob*(1 - agent_list_a[temp_tuple[0][k]].threat)                   
            win_list_a[i] += (1-prob)*agent_list_e[j-1].value
            
    stra_a_new_best = stra_a_new[np.argmax(win_list_a)].astype(int)
    
    AssignList_a1_new = []
    for i in range(len(agent_list_a_new)):
        AssignList_a1_new.append(''.join([str(agent_num_list_a_new[i]),' -> ',str(stra_a_new_best[i])]))
    return AssignList_a1_new
    
    
    
    
    
    