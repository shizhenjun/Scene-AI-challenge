#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include "HSL.cpp"
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
using namespace boost::filesystem;

vector<string> getFiles(const path &dirPath, map<string, vector<string>> &files) {
    vector<string> ret;
    directory_iterator end_itr;
    for (directory_iterator itr(dirPath); itr != end_itr; itr++) {
        if (is_directory(itr->status())) {
            string className = itr->path().stem().string();
            files[className] = getFiles(itr->path(), files);
        } else {
            string imageName = itr->path().stem().string();
            char findChar = '_';
            if (count(imageName.begin(), imageName.end(), findChar) == 1) {
                ret.push_back(itr->path().string());
            }
        }
    }
    return ret;
}

void ExpandImageSample(map<string, vector<string>> &files) {
    for (map<string, vector<string>>::iterator it = files.begin(); it != files.end(); it++) {
        cout << "---------------" << it->first << "------------------" << endl;
        vector<string> tmp = it->second;
        for (vector<string>::iterator it2 = tmp.begin(); it2 != tmp.end(); it2++) {
            //cout << (*it2) << endl;
            Mat srcImage = imread(*it2);
            HSL hsl;
            for (int j = 0; j < 2; j++) {
                Mat newImage;
                int index = rand() % 6;
                string newImageName = *it2;
                hsl.channels[index].hue += (rand() % 20 - 10);
                hsl.channels[index].saturation += (rand() % 20 - 10);
                hsl.channels[index].brightness += (rand() % 20 - 10);
                hsl.adjust(srcImage, newImage);
                int ret_index = newImageName.find(".jpg", 0);
                newImageName.insert(ret_index, 1, j + 1 + '0');
                newImageName.insert(ret_index, 1, '_');
                cout << newImageName << endl;
                imwrite(newImageName, newImage);
            }
        }
    }
}

int main() {
    srand((unsigned) time(0));
    const path dirPath("/mnt/hgfs/DataSet/train");
    map<string, vector<string>> files;
    vector<string> ret = getFiles(dirPath, files);
    ExpandImageSample(files);
    return 0;
}

