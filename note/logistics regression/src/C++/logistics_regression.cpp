#include "logistics_regression.h"
#include <stdlib.h>
#include <iostream>

double test[][3] = {
    -1.51004,
    6.061992,
    0,
    -1.07663,
    -3.18188,
    1,
    1.821096,
    10.28399,
    0,
    3.010150,
    8.401766,
    1,
    -1.09945,
    1.688274,
    1,
    -0.83487,
    -1.73386,
    1,
    -0.84663,
    3.849075,
    1,
    1.400102,
    12.62878,
    0,
    1.752842,
    5.468166,
    1,
    0.078557,
    0.059736,
    1,
    0.089392,
    -0.71530,
    1,
    1.825662,
    12.69380,
    0,
    0.197445,
    9.744638,
    0,
    0.126117,
    0.922311,
    1,
    -0.67979,
    1.220530,
    1,
    0.677983,
    2.556666,
    1,
    0.761349,
    10.69386,
    0,
    -2.16879,
    0.143632,
    1,
    1.388610,
    9.341997,
    0,
    0.317029,
    14.73902,
    0,

};
main()
{
    int test_index = 0;
    double forecast = 0;
    logisticsRegression lr("../../data/train.d");

    lr.train(0.001, 100000);

    for (test_index = 0; test_index < 20; test_index++) {
        forecast = lr.sigmod_function(test[test_index][0] * lr.w(0, 0) + test[test_index][1] * lr.w(1, 0) + lr.w(2, 0));
        printf("forecast = %f real = %f error = %f\r\n", forecast, test[test_index][2], forecast - test[test_index][2]);
    }
}