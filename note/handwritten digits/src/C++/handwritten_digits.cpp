#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <ctime>

using Eigen::MatrixXd;

class label_data {
private:
    uint32_t fread_w(FILE *__restrict __stream)
    {
        uint32_t rlt = 0;
        uint8_t i = 0;
        int ch = 0;

        for (i = 0; i < 4; i++) {
            ch = fgetc(__stream);
            if (ch != EOF) {
                rlt <<= 8;
                rlt |= ch;
            }
        }
        return rlt;
    }

public:
    uint32_t magic_number;
    uint32_t items_number;

    uint8_t *label;

    label_data(const char *file_path)
    {
        FILE *fp = fopen(file_path, "r");
        uint32_t i = 0;

        magic_number = fread_w(fp);

        //printf("magic_number = 0x%x\r\n", magic_number);

        if (magic_number != 0x00000801) {
            printf("data set error\r\n");
            exit(0);
        }

        items_number = fread_w(fp);

        //printf("items_number = %d\r\n", items_number);

        label = new uint8_t[items_number];
        uint8_t tmp = 0;
        for (i = 0; i < items_number; i++) {
            tmp = fgetc(fp);
            label[i] = tmp;
        }
    }

    void display_label(uint16_t label_index)
    {
        uint8_t *label_fp = NULL;

        label_fp = label + label_index;

        printf("label[%d]=%d\r\n", label_index, *label_fp);
    }

    uint8_t *get_label(uint16_t label_index)
    {
        uint8_t *label_fp = NULL;

        label_fp = label + label_index;
        return label_fp;
    }
};

class image_data {
private:
    uint32_t fread_w(FILE *__restrict __stream)
    {
        uint32_t rlt = 0;
        uint8_t i = 0;
        int ch = 0;

        for (i = 0; i < 4; i++) {
            ch = fgetc(__stream);
            if (ch != EOF) {
                rlt <<= 8;
                rlt |= ch;
            }
        }
        return rlt;
    }

public:
    uint32_t magic_number;
    uint32_t image_number;
    uint32_t rows_number;
    uint32_t columns_number;

    uint8_t *image;

    image_data(const char *file_path)
    {
        FILE *fp = fopen(file_path, "r");
        uint32_t i = 0;

        magic_number = fread_w(fp);

        //printf("magic_number = 0x%x\r\n", magic_number);

        if (magic_number != 0x00000803) {
            printf("data set error\r\n");
            exit(0);
        }

        image_number = fread_w(fp);
        rows_number = fread_w(fp);
        columns_number = fread_w(fp);

        //printf("image size = %d\r\nrows_number = %d\r\ncolumns_number = %d\r\n", image_number, rows_number, columns_number);

        image = new uint8_t[image_number * rows_number * columns_number];
        uint8_t tmp = 0;
        for (i = 0; i < image_number * rows_number * columns_number; i++) {
            tmp = fgetc(fp);
            image[i] = tmp;
        }
    }

    void display_pic(uint16_t pic_index)
    {
        uint32_t i = 0, j = 0;
        uint8_t *image_fp = NULL;

        image_fp = image + pic_index * columns_number * rows_number;

        for (i = 0; i < rows_number; i++) {
            for (j = 0; j < columns_number; j++) {
                if (*(image_fp + i * columns_number + j) != 0) {
                    printf("0x%2x,", *(image_fp + i * columns_number + j));
                } else {
                    printf("     ");
                }
            }
            printf("\r\n");
        }
    }

    uint8_t *get_pic(uint16_t pic_index)
    {
        uint8_t *image_fp = NULL;

        image_fp = image + pic_index * columns_number * rows_number;

        return image_fp;
    }
};

void math_lib_test(void)
{
    MatrixXd a(2, 2);
    MatrixXd b(2, 2);
    MatrixXd c(2, 2);

    a(0, 0) = 1;
    b(0, 0) = 1;
    a(0, 1) = 2;
    b(0, 1) = 2;
    a(1, 0) = 3;
    b(1, 0) = 3;
    a(1, 1) = 4;
    b(1, 1) = 4;

    c = a * b;

    printf("[[%f,%f][%f,%f]] * [[%f,%f][%f,%f]] = [[%f,%f][%f,%f]\r\n]",
           a(0, 0), a(0, 1), a(1, 0), a(1, 1),
           b(0, 0), b(0, 1), b(1, 0), b(1, 1),
           c(0, 0), c(0, 1), c(1, 0), c(1, 1));

    c = a.transpose();

    printf("[[%f,%f][%f,%f]].transpose = [[%f,%f][%f,%f]\r\n]",
           a(0, 0), a(0, 1), a(1, 0), a(1, 1),
           c(0, 0), c(0, 1), c(1, 0), c(1, 1));
}

class trainer {
public:
    uint32_t inode;
    uint32_t hnode;
    uint32_t onode;
    double learnRate;

    //result
    MatrixXd wih;
    MatrixXd who;

    trainer(uint32_t _inode, uint32_t _hnode, uint32_t _onode, double _learnRate)
    {
        uint32_t i = 0, j = 0;
        inode = _inode;
        hnode = _hnode;
        onode = _onode;
        learnRate = _learnRate;

        /* wih and who are trainer's outputs*/
        /* create wih and who space */
        wih = MatrixXd::Zero(hnode, inode);
        who = MatrixXd::Zero(onode, hnode);

        /* random initial w */
        std::default_random_engine e(time(0));
        std::normal_distribution<double> n(0, 0.07);

        for (i = 0; i < (int)wih.rows(); i++) {
            for (j = 0; j < (int)wih.cols(); j++) {
                wih(i, j) = n(e);
            }
        }
        for (i = 0; i < (int)who.rows(); i++) {
            for (j = 0; j < (int)who.cols(); j++) {
                who(i, j) = n(e);
            }
        }
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

    uint8_t forecast(uint8_t *data)
    {
        /* forward-propagating calculate error */
        uint32_t i = 0, j = 0;
        uint8_t rlt = 0;
        double max = 0;
        /* set *data into martix data structure */
        MatrixXd dataM = MatrixXd::Zero(inode, 1);

        for (i = 0; i < inode; i++) {
            dataM(i, 0) = (double)((double)data[i] / (double)255.0 * (double)0.99) + (double)0.01;
        }

        MatrixXd a1 = MatrixXd::Zero(hnode, 1);
        MatrixXd a2 = MatrixXd::Zero(onode, 1);
        /* first layer propagating */
        a1 = wih * dataM;

        /* active */
        for (i = 0; i < a1.rows(); i++) {
            for (j = 0; j < a1.cols(); j++) {
                a1(i, j) = sigmod_function(a1(i, j));
            }
        }

        /* second layer propagating */
        a2 = who * a1;

        /* active */
        for (i = 0; i < a2.rows(); i++) {
            for (j = 0; j < a2.cols(); j++) {
                a2(i, j) = sigmod_function(a2(i, j));
                if (a2(i, j) > max) {
                    max = a2(i, j);
                    rlt = i;
                }
            }
        }
        return rlt;
    }

    /* study a data */
    double train(uint8_t *data, uint8_t *label)
    {
        double loss = 0;
        /* forward-propagating calculate error */
        uint32_t i = 0, j = 0;

        /* set *data into martix data structure */
        MatrixXd dataM = MatrixXd::Zero(inode, 1);

        for (i = 0; i < inode; i++) {
            dataM(i, 0) = (double)((double)data[i] / (double)255.0 * (double)0.99) + (double)0.01;
        }

        MatrixXd a1 = MatrixXd::Zero(hnode, 1);
        MatrixXd a2 = MatrixXd::Zero(onode, 1);
        /* first layer propagating */
        a1 = wih * dataM;

        /* active */
        for (i = 0; i < a1.rows(); i++) {
            for (j = 0; j < a1.cols(); j++) {
                a1(i, j) = sigmod_function(a1(i, j));
            }
        }

        /* second layer propagating */
        a2 = who * a1;

        /* active */
        for (i = 0; i < a2.rows(); i++) {
            for (j = 0; j < a2.cols(); j++) {
                a2(i, j) = sigmod_function(a2(i, j));
            }
        }

        /*BP*/

        /* set *label into martix data structure */
        MatrixXd labelM = MatrixXd::Zero(onode, 1);

        /* set label format into output format */
        for (i = 0; i < onode; i++) {
            labelM(i, 0) = 0.01;
        }
        labelM(*label, 0) = 0.99;

        /*  layer 2 error */
        MatrixXd delta2 = MatrixXd::Zero(onode, 1);

        for (i = 0; i < delta2.rows(); i++) {
            for (j = 0; j < delta2.cols(); j++) {
                delta2(i, j) = labelM(i, j) - a2(i, j);
                loss += delta2(i, j);
            }
        }

        /*  layer 1 error */
        MatrixXd delta1 = MatrixXd::Zero(hnode, 1);

        delta1 = who.transpose() * delta2;

        /* cal dwih */
        MatrixXd dwih = MatrixXd::Zero(hnode, inode);
        MatrixXd dwho = MatrixXd::Zero(onode, hnode);

        for (i = 0; i < delta2.rows(); i++) {
            for (j = 0; j < delta2.cols(); j++) {
                delta2(i, j) = delta2(i, j) * a2(i, j) * (1 - a2(i, j));
            }
        }

        dwho = delta2 * a1.transpose();

        //cal dwho
        for (i = 0; i < delta1.rows(); i++) {
            for (j = 0; j < delta1.cols(); j++) {
                delta1(i, j) = delta1(i, j) * a1(i, j) * (1 - a1(i, j));
            }
        }

        dwih = delta2 * dataM.transpose();

        //adjust wih and who
        for (i = 0; i < dwih.rows(); i++) {
            for (j = 0; j < dwih.cols(); j++) {
                wih(i, j) = wih(i, j) + dwih(i, j) * learnRate;
            }
        }
        for (i = 0; i < dwho.rows(); i++) {
            for (j = 0; j < dwho.cols(); j++) {
                who(i, j) = who(i, j) + dwho(i, j) * learnRate;
            }
        }
        printf("%f,%f ", wih(0, 0), who(0, 0));
        return loss;
    }
};

int main()
{
    uint32_t i = 0, tim = 0;
    uint32_t epochs = 2;
    uint8_t label = 0, forcast = 0;

    image_data *train_image = new image_data("./train-images-idx3-ubyte");
    image_data *test_image = new image_data("./t10k-images-idx3-ubyte");

    label_data *train_label = new label_data("./train-labels-idx1-ubyte");
    label_data *test_label = new label_data("./t10k-labels-idx1-ubyte");

    //math_lib_test();

    trainer *t = new trainer(28 * 28, 200, 10, 0.05);
    for (tim = 0; tim < epochs; tim++) {
        for (i = 0; i < train_image->image_number; i++) {
            printf("loss=%lf %d/%d(%d)\r\n", t->train(train_image->get_pic(i), train_label->get_label(i)), i, train_image->image_number, tim);
        }
    }

    uint32_t right_num = 0;
    for (i = 0; i < test_image->image_number; i++) {
        label = *test_label->get_label(i);
        forcast = t->forecast(test_image->get_pic(i));
        if (label == forcast) {
            right_num++;
        } else {
            printf("correct = %d , forecast = %d error\r\n", label, forcast);
        }
    }

    printf("correct = %d , error = %d rate = %lf\r\n", right_num, test_image->image_number - right_num, (double)right_num / (double)test_image->image_number);
}