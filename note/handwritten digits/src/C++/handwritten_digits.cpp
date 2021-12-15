#include <iostream>

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

int main()
{
    image_data *train_image = new image_data("./train-images-idx3-ubyte");
    image_data *test_image = new image_data("./t10k-images-idx3-ubyte");

    label_data *train_label = new label_data("./train-labels-idx1-ubyte");
    label_data *test_label = new label_data("./t10k-labels-idx1-ubyte");

    train_image->display_pic(0);
    train_label->display_label(0);
}