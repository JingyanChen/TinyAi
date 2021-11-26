#include "logistics_regression.h"
#include <Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;

int main() {
  FILE *fp;
  int ch;

  if ((fp = fopen("../../data/data.d", "r")) == NULL) {
    printf("The file can not be opened.\n");
    exit(1); //结束程序的执行
  }

  ch = fgetc(fp);
  while (ch != EOF) {
    putchar(ch);
    ch = fgetc(fp);
  }

  fclose(fp);

  MatrixXd m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << m << std::endl;
}