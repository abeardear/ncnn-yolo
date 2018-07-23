// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "net.h"

struct Object{
    cv::Rect rec;
    int class_id;
    float prob;
};

const char* class_names[] = {"background",
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"};

struct BBoxRect
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    int label;
};

static inline float intersection_area(const BBoxRect& a, const BBoxRect& b)
{
    if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax || a.ymax < b.ymin)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
    float inter_height = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);

    return inter_width * inter_height;
}

template <typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores, int left, int right)
{
    int i = left;
    int j = right;
    float p = scores[(left + right) / 2];

    while (i <= j)
    {
        while (scores[i] > p)
            i++;

        while (scores[j] < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(datas[i], datas[j]);
            std::swap(scores[i], scores[j]);

            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(datas, scores, left, j);

    if (i < right)
        qsort_descent_inplace(datas, scores, i, right);
}

template <typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores)
{
    if (datas.empty() || scores.empty())
        return;

    qsort_descent_inplace(datas, scores, 0, scores.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<BBoxRect>& bboxes, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = bboxes.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        const BBoxRect& r = bboxes[i];

        float width = r.xmax - r.xmin;
        float height = r.ymax - r.ymin;

        areas[i] = width * height;
    }

    for (int i = 0; i < n; i++)
    {
        const BBoxRect& a = bboxes[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const BBoxRect& b = bboxes[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
//             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static int decoder(ncnn::Mat& bottom_top_blob, std::vector<Object>& objects, int img_h, int img_w){
    // bottom_top_blob:input [15,15,30]
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int num_box = 2;
    int num_class = 20;
    float cell_size = 1./15.;
    float contain_threshold = 0.1;
    float confidence_threshold = 0.1;
    float nms_threshold = 0.5;
    std::vector<BBoxRect> all_box_rects;
    std::vector<float> all_box_scores;
    for (int p=0; p<num_box; ++p){
        const float* xptr = bottom_top_blob.channel(p*5);
        const float* yptr = bottom_top_blob.channel(p*5+1);
        const float* wptr = bottom_top_blob.channel(p*5+2);
        const float* hptr = bottom_top_blob.channel(p*5+3);
        const float* contain_ptr = bottom_top_blob.channel(p*5+4);
        ncnn::Mat scores(w, h, num_class, (void*)((const float*)bottom_top_blob.channel(10))); //two box's class is same in yolo v1

        for(int i=0; i<h; ++i){
            for(int j=0; j<w; ++j){
                float contain_score = contain_ptr[0];
                // printf("%f", contain_score);
                if(contain_score > contain_threshold){
                    
                    float box_cx = xptr[0] * cell_size + j/15.;
                    float box_cy = yptr[0] * cell_size + i/15.;
                    float box_w = wptr[0];
                    float box_h = hptr[0];
                    // convert to xmin,ymin,xmax,ymax
                    float box_xmin = box_cx - box_w * 0.5f;
                    float box_ymin = box_cy - box_h * 0.5f;
                    float box_xmax = box_cx + box_w * 0.5f;
                    float box_ymax = box_cy + box_h * 0.5f;

                    int class_index = 0;
                    float class_score = 0.f;
                    for (int q = 0; q < num_class; q++)
                    {
                        float score = scores.channel(q).row(i)[j];
                        if (score > class_score)
                        {
                            class_index = q;
                            class_score = score;
                        }
                    }
                    float confidence = contain_score * class_score;
                    if (confidence >= confidence_threshold)
                    {
                        BBoxRect c = { box_xmin, box_ymin, box_xmax, box_ymax, class_index };
                        all_box_rects.push_back(c);
                        all_box_scores.push_back(confidence);
                    }
                }
                xptr++;
                yptr++;
                wptr++;
                hptr++;
                contain_ptr++;
            }
        }
    }
    // global sort inplace
    qsort_descent_inplace(all_box_rects, all_box_scores);
    // apply nms
    std::vector<int> picked;
    nms_sorted_bboxes(all_box_rects, picked, nms_threshold);
    // select
    std::vector<BBoxRect> bbox_rects;
    std::vector<float> bbox_scores;

    for (int i = 0; i < (int)picked.size(); i++)
    {
        int z = picked[i];
        bbox_rects.push_back(all_box_rects[z]);
        bbox_scores.push_back(all_box_scores[z]);
    }
    // fill result
    int num_detected = bbox_rects.size();
    for (int i = 0; i < num_detected; i++)
    {
        const BBoxRect& r = bbox_rects[i];
        Object object;
        float score = bbox_scores[i];
        // float* outptr = result.row(i);

        // outptr[0] = r.label + 1;// +1 for prepend background class
        // outptr[1] = score;
        // outptr[2] = r.xmin;
        // outptr[3] = r.ymin;
        // outptr[4] = r.xmax;
        // outptr[5] = r.ymax;
        object.class_id = r.label + 1;
        object.prob = score;
        object.rec.x = r.xmin * img_w;
        object.rec.y = r.ymin * img_h;
        object.rec.width = r.xmax * img_w - object.rec.x;
        object.rec.height = r.ymax * img_h - object.rec.y;
        objects.push_back(object);
    }

    return 0;

}

static int detect_yolo(cv::Mat& raw_img, float show_threshold)
{
    ncnn::Net yolo1;
    /*
     * model is  converted from https://github.com/chuanqi305/MobileNet-SSD
     * and can be downloaded from https://drive.google.com/open?id=0ByaKLD9QaPtucWk0Y0dha1VVY0U
     */
    int img_h = raw_img.size().height;
    int img_w = raw_img.size().width;
    yolo1.load_param("yolo1.param");
    yolo1.load_model("yolo1.bin");
    int input_size = 448;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(raw_img.data, ncnn::Mat::PIXEL_RGB, raw_img.cols, raw_img.rows, input_size, input_size);

    const float mean_vals[3] = {123.f, 117.f, 104.f};
    const float norm_vals[3] = {1.0/255.,1.0/255.,1.0/255.};
    // printf("%f %f %f\n", norm_vals[0], norm_vals[1], norm_vals[2]);
    in.substract_mean_normalize(mean_vals, norm_vals);
    // in.substract_mean_normalize(0, norm_vals);
    // printf("%d %d %d\n", in.w, in.h, in.c);
    // const float* data = in.channel(0);
    // for(int i=0; i<3; ++i){
    //     for(int j=0; j<10; ++j){
    //         printf("%f ", data[0]);
    //         if(j==14)
    //             printf("\n");
    //         data++;
    //     }
    // }

    ncnn::Mat out;

    ncnn::Extractor ex = yolo1.create_extractor();
    ex.set_light_mode(true);
    //ex.set_num_threads(4);
    ex.input("data", in);
    ex.extract("Sigmoid_1",out);


    // printf("%d %d %d\n", out.w, out.h, out.c);

    // const float* xptr = out.channel(4);
    // for(int i=0; i<out.h; ++i){
    //     for(int j=0; j<out.w; ++j){
    //         printf("%f ", xptr[0]);
    //         if(j==14)
    //             printf("\n");
    //         xptr++;
    //     }
    // }
    std::vector<Object> objects;
    decoder(out,objects,img_h,img_w);
    printf("ok\n");
    // for (int iw=0;iw<result.h;iw++)
    // {
    //     Object object;
    //     const float *values = result.row(iw);
    //     object.class_id = values[0];
    //     object.prob = values[1];
    //     object.rec.x = values[2] * img_w;
    //     object.rec.y = values[3] * img_h;
    //     object.rec.width = values[4] * img_w - object.rec.x;
    //     object.rec.height = values[5] * img_h - object.rec.y;
    //     objects.push_back(object);
    // }
    printf("%d\n", objects.size());
    for(int i = 0;i<objects.size();++i)
    {
        Object object = objects.at(i);
        printf("%f\n",object.prob);
        printf("%d,%d,%d,%d\n",object.rec.x,object.rec.y,object.rec.width,object.rec.height);
        if(object.prob > show_threshold)
        {
            cv::rectangle(raw_img, object.rec, cv::Scalar(255, 0, 0));
            std::ostringstream pro_str;
            pro_str<<object.prob;
            std::string label = std::string(class_names[object.class_id]) + ": " + pro_str.str();
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(raw_img, cv::Rect(cv::Point(object.rec.x, object.rec.y- label_size.height),
                                  cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), CV_FILLED);
            cv::putText(raw_img, label, cv::Point(object.rec.x, object.rec.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
    }
    cvtColor(raw_img,raw_img,CV_RGB2BGR);
    cv::imwrite("result.jpg",raw_img);
    // cv::waitKey();

    return 0;
}

int main(int argc, char** argv)
{
    float show_threshold = 0.2;
    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    cvtColor(m, m, CV_BGR2RGB);// our model trained on RGB img
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    detect_yolo(m,show_threshold);

    return 0;
}
