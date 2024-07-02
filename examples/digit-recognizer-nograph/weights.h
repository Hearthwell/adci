#ifndef DIGIT_RECOGNIZER_WEIGHTS_H
#define DIGIT_RECOGNIZER_WEIGHTS_H

extern float conv2D_filter[32][3][3][1];
extern float conv2D_bias[32];

extern float mult_weights[32];
extern float add_weights[32];

extern float conv2D_filter_1[64][3][3][32];
extern float conv2D_bias_1[64];

extern float conv2D_filter_2[128][3][3][64];
extern float conv2D_bias_2[128];

extern float fullyconnected_weights[384][6272];
extern float fullyconnected_bias[384];

extern float fullyconnected_weights_1[10][384];
extern float fullyconnected_bias_1[10];

#endif //DIGIT_RECOGNIZER_WEIGHTS_H