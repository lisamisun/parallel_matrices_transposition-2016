#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define MAX 100
#define MIN 1
#define SIZE_FOR_CACHE 8 // размерность матрицы, помещающейся в кэш
#define SIZE_FOR_MULT 32

int main(int argc, char **argv)
{
    srand(time(NULL));

    int sizeMatrix, realMatrixSize; // размер матрицы и ее настоящий размер в программе
    sscanf(argv[1], "%d", &sizeMatrix);

    double *matrix; // исходная матрица - непрерывный блок памяти
    double *transposedMatrix; // транспонированная матрица - непрерывный блок памяти

    int numTasks, num, newNumTasks, sqrtNumTasks;
    MPI_Init(&argc, &argv); // инициалзация
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks); // процессов в одном коммутаторе
    MPI_Comm_rank(MPI_COMM_WORLD, &num); // номер процесса

    MPI_Barrier(MPI_COMM_WORLD); // ждём инициализации всех процессов

    MPI_Datatype newRow, rowByRow; // новые пользовательские типы

    double startTime, processingTime, maxTime; // время транпонирования

    newNumTasks = (int)sqrt(numTasks) * (int)sqrt(numTasks); // квадратная сетка из процессов
    sqrtNumTasks = (int)sqrt(numTasks); // в одной строке сетки

    MPI_Request *requests; // для индентификации отправки
    requests = (MPI_Request *)malloc(sqrtNumTasks * sqrtNumTasks * sizeof(*requests));

    MPI_Status *statuses; // статус завершения
    statuses = (MPI_Status *)malloc(sqrtNumTasks * sqrtNumTasks * sizeof(*statuses));

    if (num >= newNumTasks) { // завершаем процессы, которые не дают сетку
        MPI_Finalize();
        return 0;
    }

    int sharedMatrixSize; // размер блока матрицы, пересылаемой одному процессу
    if (sizeMatrix % sqrtNumTasks == 0) {
        sharedMatrixSize = sizeMatrix / sqrtNumTasks;
    } else {
        sharedMatrixSize = sizeMatrix / sqrtNumTasks + 1;
    }

    int cacheBlocks; // сколько раз матрица, помещающаяся целиком в кэше, должна поместиться в sharedMatrixSize
    if (sharedMatrixSize % SIZE_FOR_MULT == 0) {
        cacheBlocks = sharedMatrixSize / SIZE_FOR_MULT;
    } else {
        cacheBlocks = sharedMatrixSize / SIZE_FOR_MULT + 1;
    }

    sharedMatrixSize = cacheBlocks * SIZE_FOR_MULT; // новый размер блока матрицы, пересылаемой одному процессу

    cacheBlocks *= 4; // количество блоков размера SIZE_FOR_CACHE

    realMatrixSize = sharedMatrixSize * sqrtNumTasks; // на самом деле размер у матрицы в программе такой

    if (num == 0) { // в основном процессе генерируем матрицу
        matrix = (double *)calloc(realMatrixSize * realMatrixSize, sizeof(*matrix));
        if (matrix == NULL) {
            printf("Can't create matrix.");
        }

        transposedMatrix = (double *)calloc(realMatrixSize * realMatrixSize, sizeof(*transposedMatrix));
        if (transposedMatrix == NULL) {
            printf("Can't create matrix.");
        }

        for (int i = 0; i < sizeMatrix; i++) { // заполняем матрицу случайными значениями, все лишнее - нули
            for (int j = 0; j < sizeMatrix; j++) {
                *(matrix + i * sizeMatrix + j) = (double)rand() / RAND_MAX * (MAX - MIN) + MIN; // генерация случайного числа - только для реальной части матрицы
            }
        }

        printf("matrix in:\n");
        for (int i = 0; i < sizeMatrix; i++) { // выводим входную матрицу
            for (int j = 0; j < sizeMatrix; j++) {
                printf("%5.3lf ", *(matrix + i * sizeMatrix + j));
            }
            printf("\n");
        }

        // отсюда начинаем считать время работы
        //startTime = MPI_Wtime();
        
        // сейчас в ОЗУ матрица хранится непрерывно построчно
        // создаем тип-шаблон для быстрого выделения из матрицы части строки
        // число элементов = sharedMatrixSize
        // элементы расположены в памяти с шагом realMatrixSize
        MPI_Type_vector(sharedMatrixSize, sharedMatrixSize, realMatrixSize, MPI_DOUBLE, &rowByRow);
        MPI_Type_commit(&rowByRow); // регистрируем тип

        // теперь надо разослать блоки размером sharedMatrixSize по всем остальным процессам
        for (int i = 0; i < sqrtNumTasks; i++) {
            for (int j = 0; j < sqrtNumTasks; j++) {
            MPI_Isend(matrix + i * realMatrixSize * sharedMatrixSize + j * sharedMatrixSize, 1, rowByRow, /*ранг процесса*/ i * sqrtNumTasks + j, /*тег сообщения*/ 0, MPI_COMM_WORLD, requests + i * sqrtNumTasks + j);
            }
        }
    }

    double *matrixForProcForFree, *matrixForProc;
    matrixForProcForFree = (double *)calloc(sharedMatrixSize * sharedMatrixSize + 32, sizeof(*matrixForProcForFree)); // выделяем больше, чем нужно
    matrixForProc = (double *) (((unsigned long long)matrixForProcForFree & (~0xFF)) + 0x100); // выравниваем память

    // во всех процессах, которые еще работают, совершаем приём
    MPI_Recv(matrixForProc, sharedMatrixSize * sharedMatrixSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, statuses + num);
    
        printf("matrix in:\n");
        for (int i = 0; i < sharedMatrixSize; i++) { // выводим входную матрицу
            for (int j = 0; j < sharedMatrixSize; j++) {
                printf("%5.3lf ", *(matrixForProc + i * sharedMatrixSize + j));
            }
            printf("\n");
        }

    /*
    // теперь в блоках размером SIZE_FOR_CACHE транспонируем матрицу
    // при этом матрица делится на cacheBlocks * cacheBlocks блоков
    // тут же используем OpenMP
    //#pragma omp parallel for
    for (int bl_i = 0; bl_i < cacheBlocks; bl_i++) {
        for (int bl_j = 0; bl_j < cacheBlocks; bl_j++) {
            // внутри блока - обычное транспонирование матрицы
            for (int i = 1; i < SIZE_FOR_CACHE; i++) {
                for (int j = 0; j < i; j ++) {
                    double tmp = *(matrixForProc + i * SIZE_FOR_CACHE + j);
                    *(matrixForProc + i * SIZE_FOR_CACHE + j) = *(matrixForProc + j * SIZE_FOR_CACHE + i);
                    *(matrixForProc + j * SIZE_FOR_CACHE + i) = tmp;
                }
            }
        }
    }

    // теперь транспонируем матрицу блоками размером sharedMatrixSize * sharedMatrixSize
    // одновременно получаем транспонирование матрицы блоками размером SIZE_FOR_CACHE * SIZE_FOR_CACHE

    MPI_Datatype rowByParts, blockOfRows, matrixByBlocks;

    // создаем тип-шаблон для быстрого собирания из матрицы строки по частям
    // число элементов = SIZE_FOR_CACHE
    // элементы расположены в памяти с шагом 1
    MPI_Type_vector(cacheBlocks, SIZE_FOR_CACHE, cacheBlocks * SIZE_FOR_CACHE * SIZE_FOR_CACHE, MPI_DOUBLE, &rowByParts);
    MPI_Type_commit(&rowByParts); // регистрируем тип

    // создаем тип-шаблон для объединения строк матрицы размером sharedMatrixSize * sharedMatrixSize в блоки по SIZE_FOR_CACHE
    // число элементов = SIZE_FOR_CACHE
    // каждый элемент содержит одну строку - ячейку типа rowByParts
    // начала строк отстоят друг от друга на расстоянии SIZE_FOR_CACHE * cacheBlocks * sizeof(*matrix) байт
    MPI_Type_hvector(SIZE_FOR_CACHE, 1, SIZE_FOR_CACHE * cacheBlocks * sizeof(*matrix), rowByParts, &blockOfRows);
    MPI_Type_commit(&blockOfRows); // регистрируем тип

    // создаем тип-шаблон для поблочного представления матрицы размером sharedMatrixSize * sharedMatrixSize
    // число элементов = cacheBlocks
    // каждый элемент содержит один блок - ячейку типа blockOfRows
    // начала блоков отстоят друг от друга на расстоянии SIZE_FOR_CACHE * sizeof(*matrix) байт
    MPI_Type_hvector(cacheBlocks, 1, SIZE_FOR_CACHE * sizeof(*matrix), blockOfRows, &matrixByBlocks);
    MPI_Type_commit(&matrixByBlocks); // регистрируем тип

    // посылаем полученный пользовательский тип в главный процесс
    MPI_Isend(matrixForProc, 1, matrixByBlocks, 0, 1, MPI_COMM_WORLD, requests + num);

    if (num == 0) { // в главном процессе собираем матрицу

        // создаем тип-шаблон для хранения строки матрицы размером sharedMatrixSize * sharedMatrixSize
        // число элементов = sharedMatrixSize
        // элементы расположены в памяти с шагом 1
        MPI_Type_vector(sharedMatrixSize, 1, 1, MPI_DOUBLE, &newRow);
        MPI_Type_commit(&newRow); // регистрируем тип

        MPI_Datatype partOfMatrix;
        // создаем тип-шаблон для построчной записи матрицы размером sharedMatrixSize * sharedMatrixSize в результирующую матрицу
        // число элементов = sharedMatrixSize
        // каждый элемент содержит одну строку - ячейку типа newRow
        // начала строк отстоят друг от друга на расстоянии realMatrixSize * sizeof(*transposedMatrix) байт
        MPI_Type_hvector(sharedMatrixSize, 1, realMatrixSize * sizeof(*transposedMatrix), newRow, &partOfMatrix);
        MPI_Type_commit(&partOfMatrix); // регистрируем тип

        for (int i = 0; i < sqrtNumTasks; i++) {
            for (int j = 0; j < sqrtNumTasks; j++) {
                MPI_Irecv(transposedMatrix + j * realMatrixSize * sharedMatrixSize + i * sharedMatrixSize, 1, partOfMatrix,
                          i * sqrtNumTasks + j, 1, MPI_COMM_WORLD, requests + i * sqrtNumTasks + j);
            }
        }

        MPI_Waitall(sqrtNumTasks * sqrtNumTasks, requests, statuses); // ждём завершения всех обменов
        
        MPI_Type_free(&newRow);
        MPI_Type_free(&partOfMatrix);
        // ура, мы вроде транспонировали матрицу
    }

    MPI_Type_free(&rowByParts);
    MPI_Type_free(&blockOfRows);
    MPI_Type_free(&matrixByBlocks);

    // здесь заканчиваем считать врея работы
    //processingTime = MPI_Wtime() - startTime;

    //MPI_Reduce(&processingTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (num == 0) { // в основном процессе есть транспонированная матрица
        printf("matrix out:\n");
        for (int i = 0; i < sizeMatrix; i++) { // выводим, что получилось
            for (int j = 0; j < sizeMatrix; j++) {
                printf("%5.3lf ", *(transposedMatrix + i * sizeMatrix + j));
            }
            printf("\n");
        }
    }

    /*if (num == 0) {
        printf("Resulted time: %d\n", maxTime);
    }*/ */

    // освобождаем память
    free(matrix);
    free(transposedMatrix);
    free(matrixForProcForFree);

    // завершаем все процессы
    MPI_Finalize();
    return 0;
}
