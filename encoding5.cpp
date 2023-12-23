#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv.h>
#include <dlib/serialize.h>
#include <iostream>
#include <vector>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/dnn.h>
#include <dlib/serialize.h>
#include <iostream>
#include <dlib/serialize.h>
#include <opencv2/opencv.hpp>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/statistics.h>
#include <vector>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/misc_api.h>
#include <utility>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_io.h>



using namespace dlib;
using namespace cv;
// Define the type alias for the face encoding matrix
using face_encoding_matrix = dlib::matrix<float,0,1>;

// These are template functions used to define residual blocks in the neural network architecture. A residual block is a building block used in deep neural networks for better convergence and improved performance. Here, the residual blocks are defined using the "add_prev1" and "add_prev2" layers, which add the output of the previous layer to the current layer. These residual blocks are used later in the definition of the neural network architecture.

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

// This is a template function used to define a basic building block in the neural network architecture. This block consists of two convolutional layers with batch normalization and ReLU activation functions.

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

// These are template functions that use the residual blocks and the basic building block defined earlier to create "ares" and "ares_down" blocks. These blocks are used later in the definition of the neural network architecture.

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

// These are template functions that use the "ares" and "ares_down" blocks to define the different levels in the neural network architecture. These levels are used later to create the final neural network model for face recognition.

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

std::vector<matrix<float,0,1>> get_face_encodings(const std::string& img_path){

    // Loadinng the input image
    cv::Mat img = cv::imread(img_path);

    // Converting the image to dlib format
    cv_image<bgr_pixel> dlib_img(img);

    // Loading the face detector and the face recognition model
    frontal_face_detector detector = get_frontal_face_detector();//returns a list of rectangles that indicate the position and size of the detected faces in the input image.

    // predicts the locations of 68 facial landmarks
    shape_predictor predictor;
    deserialize("/home/vineetha/Documents/Face Recognition/head/shape_predictor_68_face_landmarks.dat") >> predictor;

    anet_type net;// type alias for a neural network model used for face recognition
    //based on the ResNet-34 architecture and is trained to extract a 128-dimensional feature vector, or embedding
    deserialize("/home/vineetha/Documents/Face Recognition/head/face/dlib_face_recognition_resnet_model_v1.dat") >> net;

    // Detecting and aligning faces in the image

    //vector of matrices of RGB pixel values to store the cropped face images that are detected in input image
    std::vector<matrix<rgb_pixel>> faces; //Each element of the vector is a matrix of RGB pixel values that represents one of the detected faces in the input image
    
    // returns a vector of rectangles that indicate the position and size of the detected faces in the input image. Each rectangle is represented by a dlib::rectangle object, which stores the (x,y) coordinates of the top-left corner of the rectangle, as well as its width and height.
    std::vector<dlib::rectangle> face_rects = detector(dlib_img);//stores the coordinates of the detected faces in a vector of rectangles.

    // Print the number of faces detected
    std::cout << "Number of faces detected: " << face_rects.size() << std::endl;


    // Each element of the vector is a rectangle of the OpenCV library that represents one of the detected faces in the input image. The rectangle dimensions correspond to the height and width of the detected face, and the rectangle elements represent the coordinates of the top-left corner and the width and height of the rectangle.
    std::vector<cv::Rect> cv_face_rects;

    for (auto face_rect : face_rects)
    {
        // Step 01. Convert the dlib rectangle to a cv::Rect
        cv::Rect cv_face_rect(face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height());
        cv_face_rects.push_back(cv_face_rect);

        //Step 02: Get the facial landmarks for the face
        full_object_detection landmarks = predictor(dlib_img, face_rect);
        dlib::image_window win;
        win.set_image(dlib_img);
        win.add_overlay(dlib::render_face_detections(landmarks));
        win.wait_until_closed();

        //Step 03. Align the face to a standardized size and orientation
        matrix<rgb_pixel> face_chip;
        //extract_image_chip is used to crop and align the face to a standardized size and orientation
        //get_face_chip_details() function computes the details of the face chip, such as the location and scale of the face in the output chip, and returns them as a chip_details object.
        //extract_image_chip() function crops and aligns the input image based on the chip details and stores the result in the face_chip matrix
        extract_image_chip(dlib_img, get_face_chip_details(landmarks, 150, 0.25), face_chip);// 150 is the  o/p img size and 0.25  is the padding before cropping
        //resulting aligned face image is then added to the faces vector.
        faces.push_back(std::move(face_chip));
    }
    // Print the nuummber cropped face images that are detected in input image
    std::cout << "Number of cropped face images that are detected in input image: " << faces.size() << std::endl;

    // Generate face encoding vectors for each face
    //The input face images are represented as matrices of floating point values, where each row represents a flattened version of one face image.
    std::vector<matrix<float,0,1>> face_encodings = net(faces);

    // Print the size of the face_encodings vector to the console
    std::cout << "Number of face embeddings: " << face_encodings.size() << std::endl;

    // Print the face encoding vectors
    
    for (auto encoding : face_encodings)
    {
        std::cout << encoding << std::endl;
    }
    return face_encodings;
}
// Function to calculate the Euclidean distance between two face encodings
double calculate_distance(const face_encoding_matrix& face_encoding_1, const face_encoding_matrix& face_encoding_2)
{
    double distance = length(face_encoding_1 - face_encoding_2);
    return distance;
}
int main(){
    const std::string& img_path1 = "/home/vineetha/Documents/Face Recognition/head/face/frame_10.png";
    const std::string& img_path2 = "/home/vineetha/Documents/Face Recognition/head/face/chin_down.png";

    std::cout << "Generating the face encodings of input image(function called): " << std::endl;
    std::vector<matrix<float,0,1>> face_encodings_1 = get_face_encodings(img_path1);
    

    std::cout << "Generating the face encodings of test image(function called): " << std::endl;
    std::vector<matrix<float,0,1>> face_encodings_2 = get_face_encodings(img_path2);
    

    // Calculate the Euclidean distance between the face encoding for image 2 and all face encodings in image 1
    bool match_found = false;
    int face  = 0;
    for (const auto& face_encoding_1 : face_encodings_1) {
        double distance = calculate_distance(face_encoding_1, face_encodings_2[0]);
        std::cout << "Euclidean Distance between face: " <<face<<"and test face is :" <<distance<< std::endl;
        if (distance < 0.5) {
            match_found = true;
            // break;
        }
        face = face + 1;
    }

    // Check if a match was found and print the result to the console
    if (match_found) {
        std::cout << "Match found!" << std::endl;
    } else {
        std::cout << "No match found." << std::endl;
    }

    // double distance = 0;
    // for (int i = 0; i < face_encodings_1.size(); ++i) {
    //     distance += length(face_encodings_1[i] - face_encodings_2[i]);
    // }

    // std::cout << "Euclidean distance between face_encodings_1 and face_encodings_2: " << distance << std::endl;

    // if (distance < 0.5) {
    //         std::cout << "The two faces belong to the same person" << std::endl;
    // }
    // else {
    //     std::cout << "The two faces belong to different people" << std::endl;
    // }

    return 0;


}