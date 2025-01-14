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

#include "Viewer.h"
#include <pangolin/pangolin.h>

#include <mutex>

#include "include/Auxiliary.h"

namespace ORB_SLAM2 {

    Viewer::Viewer(System *pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking,
                   const string &strSettingPath, bool bReuse, bool isPangolinExists) :
            mpSystem(pSystem), mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpTracker(pTracking),
            mbFinishRequested(false), mbFinished(true), mbStopped(false), mbStopRequested(false) {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        float fps = fSettings["Camera.fps"];
        if (fps < 1)
            fps = 30;
        mT = 1e3 / fps;

        mImageWidth = fSettings["Camera.width"];
        mImageHeight = fSettings["Camera.height"];
        if (mImageWidth < 1 || mImageHeight < 1) {
            mImageWidth = 640;
            mImageHeight = 480;
        }

        mViewpointX = fSettings["Viewer.ViewpointX"];
        mViewpointY = fSettings["Viewer.ViewpointY"];
        mViewpointZ = fSettings["Viewer.ViewpointZ"];
        mViewpointF = fSettings["Viewer.ViewpointF"];
        mbReuse = bReuse;
        this->isPangolinExists = isPangolinExists;

        // std::string settingPath = Auxiliary::GetGeneralSettingsPath();
        // std::ifstream programData(settingPath);
        // nlohmann::json data;
        // programData >> data;
        // programData.close();

        // std::string map_input_dir = data["mapInputDir"];
        // mCloudPoints = map_input_dir + "cloud1.csv";

        // double startPointX = data["startingCameraPosX"];
        // double startPointY = data["startingCameraPosY"];
        // double startPointZ = data["startingCameraPosZ"];
        // mCurrentPosition = cv::Point3d(startPointX, startPointY, startPointZ);
        // mCurrentYaw = data["yawRad"];
        // mCurrentPitch = data["pitchRad"];
        // mCurrentRoll = data["rollRad"];

        // mNewPointsSeen = Auxiliary::getPointsFromPos(mCloudPoints, mCurrentPosition, mCurrentYaw, mCurrentPitch, mCurrentRoll, mTwc);
        // mPointsSeen = std::vector<cv::Point3d>();

        // mMovingScale = data["movingScale"];
        // mRotateScale = data["rotateScale"];
    }

    void Viewer::Run() {
        mbFinished = false;
        if (isPangolinExists) {
            pangolin::BindToContext("ORB-SLAM2: Map Viewer");
        } else {
            pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer", 1024, 768);
        }


        // 3D Mouse handler requires depth testing to be enabled
        glEnable(GL_DEPTH_TEST);

        // Issue specific OpenGl we might need
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        if (isPangolinExists) {
            pangolin::Panel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
        } else {
            pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
        }
        pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
        pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
        pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
        pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);
        pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode", mbReuse, true);
        pangolin::Var<bool> menuOpenSimulator("menu.Open Simulator", false, true);
        pangolin::Var<bool> menuReset("menu.Reset", false, false);
        pangolin::Var<bool> menuShutDown("menu.ShutDown", false, false);
        pangolin::Var<bool> menuMoveLeft("menu.Move Left", false, false);
        pangolin::Var<bool> menuMoveRight("menu.Move Right", false, false);
        pangolin::Var<bool> menuMoveDown("menu.Move Down", false, false);
        pangolin::Var<bool> menuMoveUp("menu.Move Up", false, false);
        pangolin::Var<bool> menuRotateLeft("menu.Rotate Left", false, false);
        pangolin::Var<bool> menuRotateRight("menu.Rotate Right", false, false);
        pangolin::Var<bool> menuRotateDown("menu.Rotate Down", false, false);
        pangolin::Var<bool> menuRotateUp("menu.Rotate Up", false, false);

        // Define Camera Render Object (for view / scene browsing)
        pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
                pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)
        );

        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View &d_cam = pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                .SetHandler(new pangolin::Handler3D(s_cam));

        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();

        // cv::namedWindow("ORB-SLAM2: Current Frame");

        bool bFollow = true;
        bool bLocalizationMode = mbReuse;

        while (1) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            if (!menuOpenSimulator)
            {
                mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);
            }
            else
            {
                Twc.m[0] = (float)mTwc.at<double>(0);
                Twc.m[1] = (float)mTwc.at<double>(1);
                Twc.m[2] = (float)mTwc.at<double>(2);
                Twc.m[4] = (float)mTwc.at<double>(4);
                Twc.m[5] = (float)mTwc.at<double>(5);
                Twc.m[6] = (float)mTwc.at<double>(6);
                Twc.m[8] = (float)mTwc.at<double>(8);
                Twc.m[9] = (float)mTwc.at<double>(9);
                Twc.m[10] = (float)mTwc.at<double>(10);
                Twc.m[12] = (float)mTwc.at<double>(12);
                Twc.m[13] = (float)mTwc.at<double>(13);
                Twc.m[14] = (float)mTwc.at<double>(14);
            }

            if (menuFollowCamera && bFollow) {
                s_cam.Follow(Twc);
            } else if (menuFollowCamera && !bFollow) {
                s_cam.SetModelViewMatrix(
                        pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
                s_cam.Follow(Twc);
                bFollow = true;
            } else if (!menuFollowCamera && bFollow) {
                bFollow = false;
            }

            if (menuLocalizationMode && !bLocalizationMode && !menuOpenSimulator) {
                mpSystem->ActivateLocalizationMode();
                bLocalizationMode = true;
            } else if (!menuLocalizationMode && bLocalizationMode && !menuOpenSimulator) {
                mpSystem->DeactivateLocalizationMode();
                bLocalizationMode = false;
            }

            d_cam.Activate(s_cam);

            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            // mpMapDrawer->DrawCurrentCamera(Twc);
            if (!menuOpenSimulator && (menuShowKeyFrames || menuShowGraph))
                mpMapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);
            if (menuShowPoints) {
                if (!menuOpenSimulator)
                {
                    mpMapDrawer->DrawMapPoints();
                }
                else
                {
                    mpMapDrawer->DrawMapPoints(true, mPointsSeen, mNewPointsSeen);
                }
                
            }

            pangolin::FinishFrame();

            if (!menuOpenSimulator)
            {
                if (mpFrameDrawer != nullptr){
                    cv::Mat im = mpFrameDrawer->DrawFrame();
                    cv::imshow("ORB-SLAM2: Current Frame", im);
                    cv::waitKey(mT);
                }

            }

            if (menuMoveLeft)
            {
                mPointsSeen.insert(mPointsSeen.end(), mNewPointsSeen.begin(), mNewPointsSeen.end());

                mCurrentPosition.x -= mMovingScale;

                mNewPointsSeen = Auxiliary::getPointsFromPos(mCloudPoints, mCurrentPosition, mCurrentYaw, mCurrentPitch, mCurrentRoll, mTwc);
                std::vector<cv::Point3d>::iterator it;
                for (it = mNewPointsSeen.begin(); it != mNewPointsSeen.end();)
                {
                    if (std::find(mPointsSeen.begin(), mPointsSeen.end(), *it) != mPointsSeen.end())
                    {
                        it = mNewPointsSeen.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
                menuMoveLeft = false;
            }

            if (menuMoveRight)
            {
                mPointsSeen.insert(mPointsSeen.end(), mNewPointsSeen.begin(), mNewPointsSeen.end());

                mCurrentPosition.x += mMovingScale;

                mNewPointsSeen = Auxiliary::getPointsFromPos(mCloudPoints, mCurrentPosition, mCurrentYaw, mCurrentPitch, mCurrentRoll, mTwc);
                std::vector<cv::Point3d>::iterator it;
                for (it = mNewPointsSeen.begin(); it != mNewPointsSeen.end();)
                {
                    if (std::find(mPointsSeen.begin(), mPointsSeen.end(), *it) != mPointsSeen.end())
                    {
                        it = mNewPointsSeen.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
                menuMoveRight = false;
            }

            if (menuMoveDown)
            {
                mPointsSeen.insert(mPointsSeen.end(), mNewPointsSeen.begin(), mNewPointsSeen.end());

                mCurrentPosition.y -= mMovingScale;

                mNewPointsSeen = Auxiliary::getPointsFromPos(mCloudPoints, mCurrentPosition, mCurrentYaw, mCurrentPitch, mCurrentRoll, mTwc);
                std::vector<cv::Point3d>::iterator it;
                for (it = mNewPointsSeen.begin(); it != mNewPointsSeen.end();)
                {
                    if (std::find(mPointsSeen.begin(), mPointsSeen.end(), *it) != mPointsSeen.end())
                    {
                        it = mNewPointsSeen.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
                menuMoveDown = false;
            }

            if (menuMoveUp)
            {
                mPointsSeen.insert(mPointsSeen.end(), mNewPointsSeen.begin(), mNewPointsSeen.end());

                mCurrentPosition.y += mMovingScale;

                mNewPointsSeen = Auxiliary::getPointsFromPos(mCloudPoints, mCurrentPosition, mCurrentYaw, mCurrentPitch, mCurrentRoll, mTwc);
                std::vector<cv::Point3d>::iterator it;
                for (it = mNewPointsSeen.begin(); it != mNewPointsSeen.end();)
                {
                    if (std::find(mPointsSeen.begin(), mPointsSeen.end(), *it) != mPointsSeen.end())
                    {
                        it = mNewPointsSeen.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
                menuMoveUp = false;
            }

            if (menuRotateLeft)
            {
                mPointsSeen.insert(mPointsSeen.end(), mNewPointsSeen.begin(), mNewPointsSeen.end());

                mCurrentYaw -= mRotateScale;

                mNewPointsSeen = Auxiliary::getPointsFromPos(mCloudPoints, mCurrentPosition, mCurrentYaw, mCurrentPitch, mCurrentRoll, mTwc);
                std::cout << "Current Pos: " << mCurrentPosition << ", yaw: " << mCurrentYaw << ", pitch: " << mCurrentPitch << ", roll: " << mCurrentRoll << std::endl;
                std::vector<cv::Point3d>::iterator it;
                for (it = mNewPointsSeen.begin(); it != mNewPointsSeen.end();)
                {
                    if (std::find(mPointsSeen.begin(), mPointsSeen.end(), *it) != mPointsSeen.end())
                    {
                        it = mNewPointsSeen.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
                menuRotateLeft = false;
            }

            if (menuRotateRight)
            {
                mPointsSeen.insert(mPointsSeen.end(), mNewPointsSeen.begin(), mNewPointsSeen.end());

                mCurrentYaw += mRotateScale;

                mNewPointsSeen = Auxiliary::getPointsFromPos(mCloudPoints, mCurrentPosition, mCurrentYaw, mCurrentPitch, mCurrentRoll, mTwc);
                std::vector<cv::Point3d>::iterator it;
                for (it = mNewPointsSeen.begin(); it != mNewPointsSeen.end();)
                {
                    if (std::find(mPointsSeen.begin(), mPointsSeen.end(), *it) != mPointsSeen.end())
                    {
                        it = mNewPointsSeen.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
                menuRotateRight = false;
            }

            if (menuRotateDown)
            {
                mPointsSeen.insert(mPointsSeen.end(), mNewPointsSeen.begin(), mNewPointsSeen.end());

                mCurrentPitch -= mRotateScale;

                mNewPointsSeen = Auxiliary::getPointsFromPos(mCloudPoints, mCurrentPosition, mCurrentYaw, mCurrentPitch, mCurrentRoll, mTwc);
                std::vector<cv::Point3d>::iterator it;
                for (it = mNewPointsSeen.begin(); it != mNewPointsSeen.end();)
                {
                    if (std::find(mPointsSeen.begin(), mPointsSeen.end(), *it) != mPointsSeen.end())
                    {
                        it = mNewPointsSeen.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
                menuRotateDown = false;
            }

            if (menuRotateUp)
            {
                mPointsSeen.insert(mPointsSeen.end(), mNewPointsSeen.begin(), mNewPointsSeen.end());

                mCurrentPitch += mRotateScale;

                mNewPointsSeen = Auxiliary::getPointsFromPos(mCloudPoints, mCurrentPosition, mCurrentYaw, mCurrentPitch, mCurrentRoll, mTwc);
                std::vector<cv::Point3d>::iterator it;
                for (it = mNewPointsSeen.begin(); it != mNewPointsSeen.end();)
                {
                    if (std::find(mPointsSeen.begin(), mPointsSeen.end(), *it) != mPointsSeen.end())
                    {
                        it = mNewPointsSeen.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
                menuRotateUp = false;
            }

            if (menuReset) {
                menuShowGraph = true;
                menuShowKeyFrames = true;
                menuShowPoints = true;
                menuLocalizationMode = false;
                menuOpenSimulator = false;
                if (bLocalizationMode)
                    mpSystem->DeactivateLocalizationMode();
                bLocalizationMode = false;
                bFollow = true;
                menuFollowCamera = true;
                mpSystem->Reset();
                menuReset = false;

                mCurrentPosition = cv::Point3d(0, 0, 0);
                mCurrentYaw = 0;
                mCurrentPitch = 0;
                mCurrentRoll = 0;

                mNewPointsSeen = Auxiliary::getPointsFromPos(mCloudPoints, mCurrentPosition, mCurrentYaw, mCurrentPitch, mCurrentRoll, mTwc);
                mPointsSeen = std::vector<cv::Point3d>();
            }

            if (menuShutDown) {
                mpSystem->shutdown_requested = true;
            }

            if (Stop()) {
                while (isStopped()) {
                    usleep(3000);
                }
            }

            if (CheckFinish())
                break;
        }

        SetFinish();
    }

    void Viewer::RequestFinish() {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    bool Viewer::CheckFinish() {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void Viewer::SetFinish() {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
    }

    bool Viewer::isFinished() {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }

    void Viewer::RequestStop() {
        unique_lock<mutex> lock(mMutexStop);
        if (!mbStopped)
            mbStopRequested = true;
    }

    bool Viewer::isStopped() {
        unique_lock<mutex> lock(mMutexStop);
        return mbStopped;
    }

    bool Viewer::Stop() {
        unique_lock<mutex> lock(mMutexStop);
        unique_lock<mutex> lock2(mMutexFinish);

        if (mbFinishRequested)
            return false;
        else if (mbStopRequested) {
            mbStopped = true;
            mbStopRequested = false;
            return true;
        }

        return false;

    }

    void Viewer::Release() {
        unique_lock<mutex> lock(mMutexStop);
        mbStopped = false;
    }

}
