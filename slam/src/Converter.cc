/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include "Converter.h"

namespace ORB_SLAM2
{
    std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat &Descriptors)
    {
        std::vector<cv::Mat> vDesc;
        vDesc.reserve(Descriptors.rows);
        int Descriptors_rows = Descriptors.rows;
        for (int j=0;j<Descriptors_rows;j++)
            vDesc.push_back(Descriptors.row(j));

        return vDesc;
    }

    g2o::SE3Quat Converter::toSE3Quat(const cv::Mat &cvT)
    {
        Eigen::Matrix<double,3,3> R;
        R << cvT.at<float>(0,0), cvT.at<float>(0,1), cvT.at<float>(0,2),
                cvT.at<float>(1,0), cvT.at<float>(1,1), cvT.at<float>(1,2),
                cvT.at<float>(2,0), cvT.at<float>(2,1), cvT.at<float>(2,2);

        Eigen::Matrix<double,3,1> t(cvT.at<float>(0,3), cvT.at<float>(1,3), cvT.at<float>(2,3));

        return g2o::SE3Quat(R,t);
    }


    cv::Mat Converter::toCvMat(const g2o::SE3Quat &SE3)
    {
        Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
        return toCvMat(eigMat);
    }

    cv::Mat Converter::toCvMat(const g2o::Sim3 &Sim3)
    {
        Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = Sim3.translation();
        double s = Sim3.scale();
        return toCvSE3(s*eigR,eigt);
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix<double,4,4> &m)
    {
        cv::Mat cvMat(4,4,CV_32F);

        cvMat.at<float>(0,0)=m(0,0);
        cvMat.at<float>(0,1)=m(0,1);
        cvMat.at<float>(0,2)=m(0,2);
        cvMat.at<float>(0,3)=m(0,3);

        cvMat.at<float>(1,0)=m(1,0);
        cvMat.at<float>(1,1)=m(1,1);
        cvMat.at<float>(1,2)=m(1,2);
        cvMat.at<float>(1,3)=m(1,3);

        cvMat.at<float>(2,0)=m(2,0);
        cvMat.at<float>(2,1)=m(2,1);
        cvMat.at<float>(2,2)=m(2,2);
        cvMat.at<float>(2,3)=m(2,3);

        cvMat.at<float>(3,0)=m(3,0);
        cvMat.at<float>(3,1)=m(3,1);
        cvMat.at<float>(3,2)=m(3,2);
        cvMat.at<float>(3,3)=m(3,3);

        return cvMat.clone();
    }


    cv::Mat Converter::toCvMat(const Eigen::Matrix3d &m)
    {
        cv::Mat cvMat(3,3,CV_32F);

        cvMat.at<float>(0,0)=m(0,0);
        cvMat.at<float>(0,1)=m(0,1);
        cvMat.at<float>(0,2)=m(0,2);


        cvMat.at<float>(1,0)=m(1,0);
        cvMat.at<float>(1,1)=m(1,1);
        cvMat.at<float>(1,2)=m(1,2);

        cvMat.at<float>(2,0)=m(2,0);
        cvMat.at<float>(2,1)=m(2,1);
        cvMat.at<float>(2,2)=m(2,2);




        return cvMat.clone();
    }


    cv::Mat Converter::toCvMat(const Eigen::Matrix<double,3,1> &m)
    {
        cv::Mat cvMat(3,1,CV_32F);

        cvMat.at<float>(0)=m(0);
        cvMat.at<float>(1)=m(1);
        cvMat.at<float>(2)=m(2);

        return cvMat.clone();
    }



    cv::Mat Converter::toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t)
    {
        cv::Mat cvMat = cv::Mat::eye(4,4,CV_32F);

        cvMat.at<float>(0,0)=R(0,0);
        cvMat.at<float>(0,1)=R(0,1);
        cvMat.at<float>(0,2)=R(0,2);

        cvMat.at<float>(1,0)=R(1,0);
        cvMat.at<float>(1,1)=R(1,1);
        cvMat.at<float>(1,2)=R(1,2);

        cvMat.at<float>(2,0)=R(2,0);
        cvMat.at<float>(2,1)=R(2,1);
        cvMat.at<float>(2,2)=R(2,2);


        cvMat.at<float>(0,3)=t(0);
        cvMat.at<float>(1,3)=t(1);
        cvMat.at<float>(2,3)=t(2);

        return cvMat.clone();
    }

    Eigen::Matrix<double,3,1> Converter::toVector3d(const cv::Mat &cvVector)
    {
        Eigen::Matrix<double,3,1> v;
        v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

        return v;
    }



    Eigen::Matrix<double,3,1> Converter::toVector3d(const cv::Point3f &cvPoint)
    {
        Eigen::Matrix<double,3,1> v;
        v << cvPoint.x, cvPoint.y, cvPoint.z;

        return v;
    }

    Eigen::Matrix<double,3,3> Converter::toMatrix3d(const cv::Mat &cvMat3)
    {
        Eigen::Matrix<double,3,3> M;

        M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
                cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
                cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

        return M;
    }


    std::vector<float> Converter::toQuaternion(const cv::Mat &M)
    {
        Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
        Eigen::Quaterniond q(eigMat);

        std::vector<float> v(4);
        v[0] = q.x();
        v[1] = q.y();
        v[2] = q.z();
        v[3] = q.w();

        return v;
    }


} //namespace ORB_SLAM
