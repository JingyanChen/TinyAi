#ifndef __LOGISTICS_REGRESSION__
#define __LOGISTICS_REGRESSION__

#include <Eigen/Dense>
#include <math.h>

using Eigen::MatrixXd;
using namespace std;
class logisticsRegression {
private:
    void loadStdFileNM(const char *__restrict file_url, int *n, int *m)
    {
        FILE *fp;
        int ch;
        int sample_num = 0;
        int dimensionality = 0;
        int onelinFinish = 0;

        if ((fp = fopen(file_url, "r")) == NULL) {
            printf("The file can not be opened.\n");
            exit(1);
        }

        /* get m & n form file */
        ch = fgetc(fp);
        while (ch != EOF) {
            if (ch == '\n') {
                sample_num++;
                onelinFinish = 1;
            }
            /* Horizontal TAB character to divider x */
            if (ch == 0x09 && onelinFinish == 0) {
                dimensionality++;
            }
            ch = fgetc(fp);
        }
        *n = dimensionality;
        *m = sample_num;

        fclose(fp);
    }

public:
    MatrixXd data;  /*(m * n) data matrix */
    MatrixXd label; /*(m * 1) label matrix */
    MatrixXd w;     /* (w0,w1,...) model parameter*/

    int n; /* dimensionality */
    int m; /* sample number*/

    /*
     * initial matrix only , user must initial matrix mannual.
     */
    logisticsRegression(int m_in, int n_in)
    {
        int i = 0;

        n = n_in;
        m = m_in;

        /* On the basis of the original characteristic dimension 
          add one more dimension for b paramer calculate */

        static MatrixXd data_instance(m, n + 1);
        static MatrixXd label_instance(m, 1);
        static MatrixXd w_instance(1, n);

        data = data_instance;
        label = label_instance;
        w = w_instance;

        /* initial last col as 1 for b calculate */
        for (i = 0; i < m; i++) {
            data_instance(i, n) = 1;
        }

        /* initial w as zero */
        for (i = 0; i < n; i++) {
            w(0, i) = 0;
        }
    }

    /* use std data file to initial object
     * file format : x11(0x09)x12(0x09) .. label(0x0d0x0a)
                     x21(0x09)x22(0x09) .. label(0x0d0x0a)
                     ...  
        This API can initial matrix by std data file             
     **/
    logisticsRegression(const char *__restrict file_url)
    {
        int index = 0, tmp = 0;
        FILE *fp;
        int ch;

        char doubledataStr[200];
        int doubluedataStrIndex = 0;

        loadStdFileNM(file_url, &n, &m);

        if ((fp = fopen(file_url, "r")) == NULL) {
            printf("The file can not be opened.\n");
            exit(1);
        }

        static MatrixXd data_instance(m, n + 1);
        static MatrixXd label_instance(m, 1);
        static MatrixXd w_instance(n + 1, 1);

        /* initial last col as 1 for b calculate */
        for (index = 0; index < m; index++) {
            data_instance(index, n) = 1;
        }

        index = 0;

        data = data_instance;
        label = label_instance;
        w = w_instance;

        //printf("%d,%d", m, n);

        ch = fgetc(fp);

        /* file format must be x1(0x09)x2(0x09)...labe(0x0a,0x0d)...
         * or this function will failed.
         */
        while (ch != EOF) {
            doubledataStr[doubluedataStrIndex] = ch;
            doubluedataStrIndex++;
            if (ch == 0x09 || ch == 0x0a) {
                doubledataStr[doubluedataStrIndex] = '\0';
                index++;
                if (index % (n + 1) == 0 && index != 0) {
                    //printf("index =%d,%f Label\n", index, atof(doubledataStr));
                    label(index / (n + 1) - 1, 0) = atof(doubledataStr);
                } else {
                    data(index / (n + 1), tmp) = atof(doubledataStr);
                    //printf("index = %d,(%d,%d)%f  Data\n", index, atof(doubledataStr), index / 3, tmp);
                    tmp++;
                    if (tmp >= n) {
                        tmp = 0;
                    }
                }

                doubluedataStrIndex = 0;
            }

            ch = fgetc(fp);
        }
        //look(label);
        fclose(fp);
    }

    void shape(MatrixXd mat)
    {
        printf("rows = %d , cols = %d", (int)mat.rows(), (int)mat.cols());
    }

    void look(MatrixXd mat)
    {
        int i = 0, j = 0;
        for (i = 0; i < (int)mat.rows(); i++) {
            printf("\n");
            for (j = 0; j < (int)mat.cols(); j++) {
                printf("%lf ", mat(i, j));
            }
        }
    }

    double sigmod_function(double x)
    {
        return 1.0 / (1 + exp(-x));
    }

    /* 
     * lamabda : study rate
     * k : loop time
     */
    void train(double lamabda, int k)
    {
        int index = 0, kx = 0;
        MatrixXd a(data.rows(), 1);
        MatrixXd loss(data.rows(), 1);
        MatrixXd dw(data.rows(), w.rows());
        double loss_sum = 0;
        int i = 0, j = 0;

        for (kx = 0; kx < k; kx++) {
            loss_sum = 0;
            a = data * w;
            /* activate */
            for (index = 0; index < a.rows(); index++) {
                a(index, 0) = sigmod_function(a(index, 0));
                //printf("%lf,%lf,%lf \r\n", label(index, 0), log(a(index, 0) / (1 - a(index, 0))), log(1 - a(index, 0)));
                loss(index, 0) = -label(index, 0) * log(a(index, 0) / (1 - a(index, 0))) - log(1 - a(index, 0));
                loss_sum += loss(index, 0);
            }

            printf("w0= %f , w1= %f , b= %f , loss sum = %f\r\n", w(0, 0), w(1, 0), w(2, 0), loss_sum);

            /* back propagation*/
            /* update w using gradient descent */
            /* batch gradient descent */

            for (i = 0; i < dw.rows(); i++) {
                for (j = 0; j < dw.cols(); j++) {
                    dw(i, j) = data(i, j) * a(i, 0) * (1 - a(i, 0)) * (((1 - label(i, 0)) / (1 - a(i, 0))) - (label(i, 0) / a(i, 0)));
                }
            }

            for (i = 0; i < dw.cols(); i++) {
                for (j = 0; j < dw.rows(); j++) {
                    w(i, 0) -= dw(j, i) * lamabda;
                }
            }
        }
    }
};

#endif