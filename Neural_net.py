# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 16:52:56 2021

@author: Fedor
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
ns = 51 # количество нейронов
los = 0

class Sequence(nn.Module):  # создать нейронную сеть
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, ns)  # слои соеденяются со слоями
        self.lstm2 = nn.LSTMCell(ns, ns)
        self.linear = nn.Linear(ns, 1)

    def forward(self, input, future = 0):  # принципы, по которым данные будут перемещаться
        outputs = []
        h_t = torch.zeros(input.size(0), ns, dtype=torch.double)
        c_t = torch.zeros(input.size(0), ns, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), ns, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), ns, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        output = torch.tensor([[1.0]], dtype=torch.double)  # корректировка перед предсказанием
        for i in range(future):    # если нужно предсказывать будущее
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
def neural_net(data, ma, w, rn1, seed):
    global los
    np.random.seed(0)   # set random seed to 0
    torch.manual_seed(seed)
    # load data and make training set
    input = torch.from_numpy(data[:1, :-1])    #с строка:, :-столбец с конца
    target = torch.from_numpy(data[:1, 1:])    #с строка:, :-столбец с начала
    test_input = torch.from_numpy(data[1:, :-1])    #до строка:, :-столбец с конца
    test_target = torch.from_numpy(data[1:, 1:])    #до строка:, :-столбец с начала
    print(ma)
    pred_n = 0
    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # задаём оптимизатор и параметры
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.2)
    pdf = PdfPages('test/predict%d+L.pdf'%w)   # создание файла pdf
    #begin to train
    for i in range(15): # задаём число эпох
        print('STEP: ', i, "ma", ma)    
        def closure():
            global los
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())  
            if loss.item() > 10:    # если плохая сходимость
                los = 1
            loss.backward()
            return loss
        optimizer.step(closure)
        if los == 1:    # обработка плохой сходимости
            los = 0
            pdf.close()
            return 0.1  # завершить работу и вернуть код ошибки.
        # начинайте прогнозировать, здесь нет необходимости отслеживать градиент
        with torch.no_grad():
            future = 5
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(30,17))
        ca = y[0]*ma 
        print('n+1' + str(ca[input.size(1):input.size(1)+1]))
        plt.title('Эпоха: ' + str(i+1) + '\nИзвестное число заражений на n день:    ' + str(int(ma)) + '    и предсказанное на n день:   '+ str(int(round(ca[input.size(1)-1:input.size(1)][0]))) + "\nИзвестное число заражений на n+1 день:    " + str(int(rn1)) + "    и предсказнное на n+1 день:    " + str(int(round(ca[input.size(1):input.size(1)+1][0]))) , fontsize=30)
        plt.xlabel('День', fontsize=30)
        plt.ylabel('Количество заражений', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.minorticks_on()
        plt.grid(which='major', alpha=0.5)
        plt.grid(which='minor', alpha=0.3, linestyle = ':')    
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), data[0, 1:]*ma, 'black', linewidth = 2.0, label = 'Реальные значения') #реальные извесные значения
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)]*ma, color, linewidth = 2.0, label = 'Предсказанные значения')
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):]*ma, color + ':', linewidth = 2.0, label = 'Предсказанные значения на будущее')
        draw(y[0], 'r')
        plt.legend(loc = 4, fontsize=30)
        pdf.savefig()
        plt.close()
        if abs(int(round(ca[input.size(1):input.size(1)+1][0])) - ma) < ma :  # отсев выбросов если ошибка больше по модулю предсказанного значения
            pred_n = int(round(ca[input.size(1):input.size(1)+1][0]))
    pdf.close() # сохранение pdf
    return pred_n
    
if __name__ == '__main__':
    
    cas = []    # число заражений для обучения
    cas1 = []   # число заражений по которым будет предсказывать
    count = 0
    with open('datapy3.csv') as File:   # получение данных из csv
     for row in csv.reader(File, delimiter=';'):
            cas.append(row[0])
            cas1.append(row[1]) 
            count += 1       
    data = np.empty((2, count), 'int64')    # данные по числу заражений для нейросети
    data[0] =  cas
    data[1] =  cas1
    data = data.astype('float64') 
    real_data = data.copy() # копия данных
   
    for L in range(10, 201, 10):    # меняем размер окна 
        error = []  # ошибка
        val_cal = []    # предсказанное
        r_d = []   # реальное известное
        mae = 0
        for J in range(0, count-L, 1):    # начальное J, по T-L, шаг
            Start = J   # начальное значение окна
            End = Start + L     # конечное значение окна
            value_calculated = 0.1
            seed = 0
            while value_calculated == 0.1:  # если будет плохая сходимость, будем перезапускать нейронную сеть с другим зерном пока не будет нормальной сходимости
                value_calculated = neural_net(data[:, Start:End]/data[0, End-1], real_data[0, End-1], J, real_data[0, End], seed)   # data - окно размером L, последний End элемент для множителя, номер для графика, будущее значения для графика, зерно
                seed += 1
            print("value_calculated: ", value_calculated)
            error.append(real_data[0, End] - value_calculated)  # запоминаем значение ошибки
            val_cal.append(value_calculated)    # запоминаем предсказанное значение
            r_d.append(real_data[0, End])   # запоминаем реальное значение
            print("error: ", error) 
        
        with open('error_%d.txt'%L,'a') as f:    # сохраняем данные
            for i in error:
                f.write('%s ' % i) 
            #f.write('\n')
        with open('valcal_%d.txt'%L,'a') as f:   # сохраняем данные
            for i in val_cal:
                f.write('%s ' % i) 
            #f.write('\n')
        with open('rd_%d.txt'%L,'a') as f:   # сохраняем данные
            for i in r_d:
                f.write('%s ' % i) 
            #f.write('\n')
        for t in range(0, len(error)):  #средняя абсолютная ошибка
            mae = mae + abs(error[t])
        mae = mae/len(error)
        # рисуем графики    
        plt.figure(figsize=(30,13))
        plt.title('Ошибка предсказания на Start...End день\nСредняя абсолютная ошибка: ' + str(mae) , fontsize=30)
        plt.xlabel('День', fontsize=20)
        plt.ylabel('Ошибка', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.minorticks_on()
        plt.grid(which='major', alpha=0.5)
        plt.grid(which='minor', alpha=0.3, linestyle = ':')
        plt.plot(np.arange(L, len(error)+L), error, 'black', linewidth = 2.0) # график ошибок
        plt.savefig('error_%d.pdf'%L)
    
    
    
        