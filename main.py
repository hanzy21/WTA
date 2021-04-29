import method
from matplotlib import pyplot as plt
import numpy as np
import time
import geatpy as ea
from prettytable import PrettyTable
from prettytable import ALL

time0 = time.time()
#参数
field_threshold = 7 #可通行区域威胁度上限            
ActionTime = 100 #初设作战时间

# 态势场初始化
field = np.zeros((1000,1000))
for i in range(field.shape[0]):
    for j in range(field.shape[1]):
        if (i>=400)and(i<=500)and(j<=600)and(j>=400):
            field[i][j] = 10
        else:
            field[i][j] = 5

#障碍物初始化
obstacle = np.zeros((1000,1000))
for i in range(obstacle.shape[0]):
    for j in range(obstacle.shape[1]):
        if (i>=400)and(i<=500)and(j>=800 or j<=200):
            obstacle[i][j] = 1
        else:
            obstacle[i][j] = 0

# 智能体初始化
AgentList_a = []
AgentList_e = []
AgentList_a.append(method.Agent((100,100),10,0.5,10))
AgentList_a.append(method.Agent((500,100),12,0.7,20))
AgentList_a.append(method.Agent((900,100),8,0.4,8))
AgentList_e.append(method.Agent((100,900),8,0.3,13))
AgentList_e.append(method.Agent((500,900),10,0.4,15))
AgentList_e.append(method.Agent((900,900),6,0.6,18))

#AgentList_a.append(method.Agent((600,100),8,0.4,10))
#AgentList_a.append(method.Agent((300,100),8,0.4,15))
#AgentList_a.append(method.Agent((400,100),8,0.4,12))
# =============================================================================
# =============================================================================

# 枚举策略集
Stra_a, Stra_e = method.strategy(AgentList_a, AgentList_e)

time2 = time.time()

print("2", time2-time0)
# 计算时间表
Timetable_a, Timetable_e, Pathtable, Impassable = method.compute_timetable(
    AgentList_a, AgentList_e, field, field_threshold, obstacle)
time3 = time.time()
print("3", time3-time2)

#获得三个矩阵
WinMatrix_a, WinMatrix_e = method.win_matrix(AgentList_a, AgentList_e, Stra_a, Stra_e)
time4 = time.time()
print("4", time4-time3)

TimeMatrix_a, TimeMatrix_e = method.time_matrix(AgentList_a, AgentList_e, Stra_a, Stra_e, Timetable_a, Timetable_e, ActionTime)
time5 = time.time()
print("5", time5-time4)

ResourceMatrix_a, ResourceMatrix_e = method.resource_matrix(AgentList_a, AgentList_e, Stra_a, Stra_e)


#三个矩阵的化简
Matrix_a1, Matrix_e1, Matrix_a2, Matrix_e2, Matrix_a3, Matrix_e3 = method.matrix_simplify(WinMatrix_a, WinMatrix_e, TimeMatrix_a, TimeMatrix_e, ResourceMatrix_a, ResourceMatrix_e)
timea = time.time()
print("a",timea-time5)

#胜率优先
print("胜率优先策略")
Matrix_a1_final, Matrix_e1_final, Retain_a1, Retain_e1 = method.dominant(Matrix_a1, Matrix_e1)
Problem1 = method.problem(Matrix_a1_final, Matrix_e1_final)
S_a1, S_e1 = method.solve_ea(Problem1,Matrix_a1_final.shape, Retain_a1, Retain_e1)
#method.solve_pure(Matrix_a1_final, Matrix_e1_final, Retain_a1, Retain_e1)
Assign_a1 = Stra_a[S_a1].astype(int)
Assign_e1 = Stra_e[S_e1].astype(int)
AssignList_a1_first = []
AssignList_e1 = []
for i in range(len(AgentList_a)):
    AssignList_a1_first.append(''.join([str(i+1),' -> ',str(Assign_a1[i])]))
for i in range(len(AgentList_e)):
    AssignList_e1.append(''.join([str(i+1),' -> ',str(Assign_e1[i])]))

stage_num_1 = 1
if not method.DesRateEnough(Assign_a1, AgentList_a, AgentList_e):

    AssignList_a1_extra = method.ExtraAssignment(Assign_a1, Assign_e1, AgentList_a, AgentList_e)
    if AssignList_a1_extra:
        print("进行分阶段目标分配")
        stage_num_1 = 2
        AssignList_a1 = ''.join(['第一阶段: ',', '.join(AssignList_a1_first),'\n 第二阶段：',', '.join(AssignList_a1_extra)])
    else:
        AssignList_a1 = ''.join(['第一阶段: ',', '.join(AssignList_a1_first)])
else:
    AssignList_a1 = ''.join(['第一阶段: ',', '.join(AssignList_a1_first)])

#时间优先
print("时间优先策略")
Matrix_a2_final, Matrix_e2_final, Retain_a2, Retain_e2 = method.dominant(Matrix_a2, Matrix_e2)
Problem2 = method.problem(Matrix_a2_final, Matrix_e2_final)
S_a2, S_e2 = method.solve_ea(Problem2,Matrix_a2_final.shape, Retain_a2, Retain_e2)
#method.solve_pure(Matrix_a2_final, Matrix_e2_final, Retain_a2, Retain_e2)
Assign_a2 = Stra_a[S_a2].astype(int)
Assign_e2 = Stra_e[S_e2].astype(int)
AssignList_a2_first = []
AssignList_e2 = []
for i in range(len(AgentList_a)):
    AssignList_a2_first.append(''.join([str(i+1),' -> ',str(Assign_a2[i])]))
for i in range(len(AgentList_e)):
    AssignList_e2.append(''.join([str(i+1),' -> ',str(Assign_e2[i])]))

stage_num_2 = 1
if not method.DesRateEnough(Assign_a2, AgentList_a, AgentList_e):
    AssignList_a2_extra = method.ExtraAssignment(Assign_a2, Assign_e2, AgentList_a, AgentList_e)
    if AssignList_a2_extra:
        print("进行分阶段目标分配")
        stage_num_2 = 2
        AssignList_a2 = ''.join(['第一阶段: ',', '.join(AssignList_a2_first),'\n 第二阶段：',', '.join(AssignList_a2_extra)])
    else:
        AssignList_a2 = ''.join(['第一阶段: ',', '.join(AssignList_a2_first)])
else:
    AssignList_a2 = ''.join(['第一阶段: ',', '.join(AssignList_a2_first)])

#均衡策略
print("均衡策略")
Matrix_a3_final, Matrix_e3_final, Retain_a3, Retain_e3 = method.dominant(Matrix_a3, Matrix_e3)
Problem3 = method.problem(Matrix_a3_final, Matrix_e3_final)
S_a3, S_e3 = method.solve_ea(Problem3,Matrix_a3_final.shape, Retain_a3, Retain_e3)
#method.solve_pure(Matrix_a3_final, Matrix_e3_final, Retain_a3, Retain_e3)
Assign_a3 = Stra_a[S_a3].astype(int)
Assign_e3 = Stra_e[S_e3].astype(int)
AssignList_a3_first = []
AssignList_e3 = []
for i in range(len(AgentList_a)):
    AssignList_a3_first.append(''.join([str(i+1),' -> ',str(Assign_a3[i])]))
for i in range(len(AgentList_e)):
    AssignList_e3.append(''.join([str(i+1),' -> ',str(Assign_e3[i])]))

stage_num_3 = 1
if not method.DesRateEnough(Assign_a3, AgentList_a, AgentList_e):
    
    AssignList_a3_extra = method.ExtraAssignment(Assign_a3, Assign_e3, AgentList_a, AgentList_e)
    if AssignList_a3_extra:
        print("进行分阶段目标分配")
        stage_num_3 = 2
        AssignList_a3 = ''.join(['第一阶段: ',', '.join(AssignList_a3_first),'\n 第二阶段：',', '.join(AssignList_a3_extra)])
    else:
        AssignList_a3 = ''.join(['第一阶段: ',', '.join(AssignList_a3_first)])
else:
    AssignList_a3 = ''.join(['第一阶段: ',', '.join(AssignList_a3_first)])


#结果表格输出

result = PrettyTable(hrules=ALL)
result.valign = 'm'
result.align = 'c'
result.field_names = ["策略目标", "阶段数","我方分配建议", "第一阶段预计成功率(%)", "第一阶段预计耗时(s)", "预计对方第一阶段响应"]
result.add_row(["成功率较高", stage_num_1, AssignList_a1, round(WinMatrix_a[S_a1, S_e1]*100,1), 
                round(TimeMatrix_a[S_a1, S_e1],1), AssignList_e1])
# =============================================================================
result.add_row(["时间较短", stage_num_2, AssignList_a2, round(WinMatrix_a[S_a2, S_e2]*100,1), 
                round(TimeMatrix_a[S_a2, S_e2],1), AssignList_e2])
result.add_row(["均衡策略", stage_num_3, AssignList_a3, round(WinMatrix_a[S_a3, S_e3]*100,1), 
                round(TimeMatrix_a[S_a3, S_e3],1), AssignList_e3])
# =============================================================================
print(result)

"""
time7 = time.time()
print("ab", time7-timea)
print("b", time7-time0)


time1 = time.time()
print("2",time3-time2)
print("3",time4-time3)
print("4",time5-time4)
print("5",time6-time5)
print("6",time1-time6)
print("1",time1-time0)
"""
#RRT绘图
'''
for i in Pathtable:
    pathx = []
    pathy = []
    for j in i:
        pathx.append(j[0])
        pathy.append(j[1])
    plt.plot(pathx,pathy,"-b")
obstaclex = []
obstacley = []
for i in range(Impassable.shape[0]):
    for j in range(Impassable.shape[1]):
        if Impassable[i][j] == 1:
            obstaclex.append(j)
            obstacley.append(i)
plt.plot(obstaclex,obstacley,",r",markersize = 0.1)
for i in AgentList_a:    
    plt.scatter(i.pos[0],i.pos[1],c='g',s=700)
for i in AgentList_e:  
    plt.scatter(i.pos[0],i.pos[1],c='c',s=700)
plt.xlim(0,Impassable.shape[0])
plt.ylim(0,Impassable.shape[1])
plt.show()
   
time1 = time.time()
print("1",time1-time0)
    '''
