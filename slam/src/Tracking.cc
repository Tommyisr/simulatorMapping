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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2 {

    Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap,
                       KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor, const bool bReuse) :
            mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(bReuse), mbVO(false), mpORBVocabulary(pVoc),
            mpKeyFrameDB(pKFDB), mpInitializer(nullptr), mpSystem(pSys),
            mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0) {
        // Load camera parameters from settings file

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float width = fSettings["Camera.width"];
        float height = fSettings["Camera.height"];

        float orig_width = fSettings["Camera.orig_width"];
        float orig_height = fSettings["Camera.orig_height"];

        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];
        fx = fx * width / orig_width;
        fy = fy * height / orig_height;
        cx = cx * width / orig_width;
        cy = cy * height / orig_height;

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;

        is_preloaded = bReuse;
        if (is_preloaded) {
            mnLastKeyFrameId = 0;
            mState = LOST;
        }
        // Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 0;
        mMaxFrames = fps;

        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if (DistCoef.rows == 5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;


        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        // Load ORB parameters

        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (sensor == System::STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (sensor == System::MONOCULAR)
            mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        cout << endl << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;
        cout << "- Reuse Map ?: " << is_preloaded << endl;
        if (sensor == System::STEREO || sensor == System::RGBD) {
            mThDepth = mbf * (float) fSettings["ThDepth"] / fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }

        if (sensor == System::RGBD) {
            mDepthMapFactor = fSettings["DepthMapFactor"];
            if (mDepthMapFactor == 0)
                mDepthMapFactor = 1;
            else
                mDepthMapFactor = 1.0f / mDepthMapFactor;
        }

    }

    void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) {
        mpLocalMapper = pLocalMapper;
    }

    void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) {
        mpLoopClosing = pLoopClosing;
    }

    void Tracking::SetViewer(Viewer *pViewer) {
        mpViewer = pViewer;
    }


    cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp) {
        mImGray = imRectLeft;
        cv::Mat imGrayRight = imRectRight;

        if (mImGray.channels() == 3) {
            if (mbRGB) {
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
            } else {
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
            }
        } else if (mImGray.channels() == 4) {
            if (mbRGB) {
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
            } else {
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
            }
        }

        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary,
                              mK, mDistCoef, mbf, mThDepth);

        Track();

        return mCurrentFrame.mTcw.clone();
    }


    cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp) {
        mImGray = imRGB;
        cv::Mat imDepth = imD;

        if (mImGray.channels() == 3) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        } else if (mImGray.channels() == 4) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        if (mDepthMapFactor != 1 || imDepth.type() != CV_32F);
        imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

        mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf,
                              mThDepth);

        Track();

        return mCurrentFrame.mTcw.clone();
    }

    cv::Mat
    Tracking::GrabImageMonocular(const cv::Mat &descriptors, std::vector<cv::KeyPoint> &keyPoints, const float cols,
                                 const float rows, const double &timestamp, const cv::Mat &im) {

        if (mImGray.channels() == 3) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        } else if (mImGray.channels() == 4) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        mCurrentFrame = Frame(descriptors, keyPoints, cols, rows, timestamp, mpORBextractorLeft, mpORBVocabulary, mK,
                              mDistCoef,
                              mbf,
                              mThDepth);
        Track(im);

        return mCurrentFrame.mTcw.clone();
    }


    cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp) {
        mImGray = im;

        if (mImGray.channels() == 3) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        } else if (mImGray.channels() == 4) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
            mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
        else
            mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf,
                                  mThDepth);

        Track();

        return mCurrentFrame.mTcw.clone();
    }

    void Tracking::Track(const cv::Mat &im) {
        if (mState == NO_IMAGES_YET) {
            mState = NOT_INITIALIZED;
        }

        mLastProcessedState = mState;

        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        if (mState == NOT_INITIALIZED) {
            if (is_preloaded) {
                mState = LOST;
                return;
            }
            if (mSensor == System::STEREO || mSensor == System::RGBD)
                StereoInitialization();
            else
                MonocularInitialization();
            if (mpFrameDrawer != nullptr) {
                mpFrameDrawer->Update(this, im);
            }

            if (mState != OK)
                return;
        } else {
            // System is initialized. Track Frame.
            bool bOK;

            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            if (!mbOnlyTracking) {
                // Local Mapping is activated. This is the normal behaviour, unless
                // you explicitly activate the "only tracking" mode.
                //mState = LOST;
                if (mState == OK) {
                    // Local Mapping might have changed some MapPoints tracked in last frame
                    CheckReplacedInLastFrame();

                    if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                        bOK = TrackReferenceKeyFrame();
                    } else {
                        bOK = TrackWithMotionModel();
                        if (!bOK)
                            bOK = TrackReferenceKeyFrame();
                    }
                } else {
                    bOK = Relocalization();
                }
            } else {
                // Only Tracking: Local Mapping is deactivated
                // BAR
                // mState = LOST;
                if (mState == LOST) {
                    bOK = Relocalization();
                } else {
                    if (!mbVO) {
                        // In last frame we tracked enough MapPoints in the map

                        if (!mVelocity.empty()) {
                            bOK = TrackWithMotionModel();
                        } else {
                            bOK = TrackReferenceKeyFrame();
                        }
                    } else {
                        // In last frame we tracked mainly "visual odometry" points.

                        // We compute two camera poses, one from motion model and one doing relocalization.
                        // If relocalization is sucessfull we choose that solution, otherwise we retain
                        // the "visual odometry" solution.

                        bool bOKMM = false;
                        bool bOKReloc = false;
                        vector<MapPoint *> vpMPsMM;
                        vector<bool> vbOutMM;
                        cv::Mat TcwMM;
                        if (!mVelocity.empty()) {
                            bOKMM = TrackWithMotionModel();
                            vpMPsMM = mCurrentFrame.mvpMapPoints;
                            vbOutMM = mCurrentFrame.mvbOutlier;
                            TcwMM = mCurrentFrame.mTcw.clone();
                        }
                        bOKReloc = Relocalization();

                        if (bOKMM && !bOKReloc) {
                            mCurrentFrame.SetPose(TcwMM);
                            mCurrentFrame.mvpMapPoints = std::move(vpMPsMM);
                            mCurrentFrame.mvbOutlier = std::move(vbOutMM);

                            if (mbVO) {
                                int N = mCurrentFrame.N;
                                for (int i = 0; i < N; i++) {
                                    auto* pMapPoint = mCurrentFrame.mvpMapPoints[i];

                                    if (pMapPoint && !mCurrentFrame.mvbOutlier[i]) {
                                        pMapPoint->IncreaseFound();
                                    }
                                }
                            }
                        } else if (bOKReloc) {
                            mbVO = false;
                        }

                        bOK = bOKReloc || bOKMM;
                    }
                }
            }


            mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // If we have an initial estimation of the camera pose and matching. Track the local map.
            if (!mbOnlyTracking) {
                if (bOK)
                    bOK = TrackLocalMap();
            } else {
                // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
                // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
                // the camera we will use the local map again.
                if (bOK && !mbVO)
                    bOK = TrackLocalMap();
            }

            if (bOK)
                mState = OK;
            else
                mState = LOST;

            // Update drawer
            if (mpFrameDrawer != nullptr) {
                mpFrameDrawer->Update(this);
            }

            // If tracking were good, check if we insert a keyframe
            if (bOK) {
                // Update motion model
                if (!mLastFrame.mTcw.empty()) {
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                    mVelocity = mCurrentFrame.mTcw * LastTwc;
                } else
                    mVelocity = cv::Mat();


                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

                // Clean temporal point matches
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    auto *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (pMP && pMP->Observations() < 1) {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] = nullptr;

                    }
                }

                // Delete temporal MapPoints
//                for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
//                     lit != lend; lit++) {
//                    MapPoint *pMP = *lit;
//                    delete pMP;
//                }
                for (auto& pMP : mlpTemporalPoints) {
                    delete pMP;
                }

                mlpTemporalPoints.clear();

                // Check if we need to insert a new keyframe
                if (NeedNewKeyFrame())
                    CreateNewKeyFrame();

                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = nullptr;

                }
            }

            // Reset if the camera get lost soon after initialization
            if (mState == LOST && mpMap->KeyFramesInMap() <= 5) {
//                if (mpMap->KeyFramesInMap() <= 5) {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
//                }
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            mLastFrame = Frame(mCurrentFrame);
        }


        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if (!mCurrentFrame.mTcw.empty() && mCurrentFrame.mpReferenceKF) {
//            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
//            mlRelativeFramePoses.emplace_back(Tcr);
//            mlpReferences.emplace_back(mpReferenceKF);
//            mlFrameTimes.emplace_back(mCurrentFrame.mTimeStamp);
//            mlbLost.emplace_back(mState == LOST);
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.emplace_back(Tcr);
            mlpReferences.emplace_back(mpReferenceKF);
            mlFrameTimes.emplace_back(mCurrentFrame.mTimeStamp);
            mlbLost.emplace_back(mState == LOST);

        }
    }

#if 0
    cv::Mat Tracking::getTransformData()
    {
            return mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
    }
#endif




    void Tracking::StereoInitialization() {
        if (mCurrentFrame.N > 500) {
            // Set Frame pose to the origin
            mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

            // Create KeyFrame
            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

            // Insert KeyFrame in the map
            mpMap->AddKeyFrame(pKFini);

            // Create MapPoints and asscoiate to KeyFrame
            for (int i = 0; i < mCurrentFrame.N; i++) {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0) {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
                    pNewMP->AddObservation(pKFini, i);
                    pKFini->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                }
            }

            cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

            mpLocalMapper->InsertKeyFrame(pKFini);

            mLastFrame = Frame(mCurrentFrame);
            mnLastKeyFrameId = mCurrentFrame.mnId;
            mpLastKeyFrame = pKFini;

            mvpLocalKeyFrames.emplace_back(pKFini);
            mvpLocalMapPoints = mpMap->GetAllMapPoints();
            mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;

            mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

            mpMap->mvpKeyFrameOrigins.emplace_back(pKFini);

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            mState = OK;
        }
    }

    void Tracking::MonocularInitialization() {

        if (!mpInitializer) {
            // Set Reference Frame
            if (mCurrentFrame.mvKeys.size() > 100) {
                mInitialFrame = Frame(mCurrentFrame);
                mLastFrame = Frame(mCurrentFrame);
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());

                std::transform(mCurrentFrame.mvKeysUn.begin(), mCurrentFrame.mvKeysUn.end(), mvbPrevMatched.begin(),
                               [](const auto& key) { return key.pt; });

                delete mpInitializer;
                mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

                std::fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;

            }

        } else {
            // Try to initialize
            if ( mCurrentFrame.mvKeys.size() <= 100) {

                delete mpInitializer;
                mpInitializer = nullptr;
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;
            }

            // Find correspondences
            ORBmatcher matcher(0.9, true);
            int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches,
                                                           100);

            // Check if there are enough correspondences
            if (nmatches < 80) { // orig was 100
                cout << __FUNCTION__ << "ORB extraction : No enough correspondesnces(<80) " << nmatches << endl;
                delete mpInitializer;
                mpInitializer = nullptr;
                return;
            }

            cv::Mat Rcw; // Current Camera Rotation
            cv::Mat tcw; // Current Camera Translation
            vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

            if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated)) {
                for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
                    if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
                        mvIniMatches[i] = -1;
                        nmatches--;
                    }
                }

                // Set Frame Poses
                mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
                cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
                Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(Tcw.rowRange(0, 3).col(3));
                mCurrentFrame.SetPose(Tcw);

                CreateInitialMapMonocular();
            }
        }
    }

    void Tracking::CreateInitialMapMonocular() {

        cout << __FUNCTION__ << ": Starting Initial Map Creation..." << endl;
        // Create KeyFrames
        auto *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        auto *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);


        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // Insert KFs in the map
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // Create MapPoints and asscoiate to keyframes
        for (size_t i = 0; i < mvIniMatches.size(); i++) {
            if (mvIniMatches[i] < 0)
                continue;

            //Create MapPoint.


            auto worldPos = cv::Mat(mvIniP3D[i]);
            auto *pMP = new MapPoint(worldPos, pKFcur, mpMap);

            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();

            //Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            //Add to Map
            mpMap->AddMapPoint(pMP);
        }

        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        // Bundle Adjustment
        cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

        Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

        // Set median depth to 1
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f / medianDepth;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
            cout << "Wrong initialization, reseting... map points(100):"
                 << pKFcur->TrackedMapPoints(1)
                 << " medianDepth = " << medianDepth << endl;
            Reset();
            return;
        }

        // Scale initial baseline

        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0, 3) += invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale points
        auto vpAllMapPoints = pKFini->GetMapPointMatches();
        for (auto *pMP : vpAllMapPoints) {
            if (pMP) {
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            }
        }


        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.emplace_back(pKFcur);
        mvpLocalKeyFrames.emplace_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpMap->mvpKeyFrameOrigins.emplace_back(pKFini);

        mState = OK;
    }


    void Tracking::CheckReplacedInLastFrame() {

        for (std::size_t i = 0; i < mLastFrame.N; ++i) {
            auto *pMP = mLastFrame.mvpMapPoints[i];
            if (pMP) {
                auto *pRep = pMP->GetReplaced();
                if (pRep) {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }


    }


    bool Tracking::TrackReferenceKeyFrame() {
        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);
        vector<MapPoint *> vpMapPointMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

        if (nmatches < 15)
            return false;

        mCurrentFrame.mvpMapPoints = std::move(vpMapPointMatches);
        mCurrentFrame.SetPose(mLastFrame.mTcw);

        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        for (std::size_t i = 0; i < mCurrentFrame.N; ++i) {
            auto* pMP = mCurrentFrame.mvpMapPoints[i];
            if (pMP) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    mCurrentFrame.mvpMapPoints[i] = nullptr;
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    --nmatches;
                } else if (pMP->Observations() > 0) {
                    ++nmatchesMap;
                }
            }
        }

        return nmatchesMap >= 10;
    }


    void Tracking::UpdateLastFrame() {
        // Update pose according to reference keyframe
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        cv::Mat Tlr = mlRelativeFramePoses.back();

        mLastFrame.SetPose(Tlr * pRef->GetPose());

        if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking)
            return;

        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vDepthIdx;
        int N = mLastFrame.N;
        vDepthIdx.reserve(N);
        for (int i = 0; i < N; i++) {
            float z = mLastFrame.mvDepth[i];
            if (z > 0) {
                vDepthIdx.emplace_back(z, i);
            }
        }

        if (vDepthIdx.empty())
            return;

        sort(vDepthIdx.begin(), vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        int nPoints = 0;
        for (const auto& depthIdx : vDepthIdx) {
            int i = depthIdx.second;

            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            if (!pMP || pMP->Observations() < 1) {
                cv::Mat x3D = mLastFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

                mLastFrame.mvpMapPoints[i] = pNewMP;
                mlpTemporalPoints.emplace_back(pNewMP);
            }

            nPoints++;

            if (depthIdx.first > mThDepth && nPoints > 100)
                break;
        }
    }


    bool Tracking::TrackWithMotionModel() {
        ORBmatcher matcher(0.9, true);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points
        UpdateLastFrame();

        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);

        // Project points seen in previous frame
        int th = (mSensor != System::STEREO) ? 15 : 7;
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

        // If few matches, uses a wider window search
        if (nmatches < 20) {
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);

            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR);
        }

        if (nmatches < 20)
            return false;

        // Optimize frame pose with all matches
        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        int N = mCurrentFrame.N;
        for (int i = 0; i < N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {

                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = nullptr;

                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        if (mbOnlyTracking) {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }

        return nmatchesMap >= 10;
    }



    bool Tracking::TrackLocalMap() {
        // We have an estimation of the camera pose and some map points tracked in the frame.
        // We retrieve the local map and try to find matches to points in the local map.

        UpdateLocalMap();

        SearchLocalPoints();

        // Optimize Pose
        Optimizer::PoseOptimization(&mCurrentFrame);
        mnMatchesInliers = 0;

        // Update MapPoints Statistics
        int N = mCurrentFrame.N;
        for (int i = 0; i < N; ++i) {
            auto& pMP = mCurrentFrame.mvpMapPoints[i];
            if (pMP) {
                if (!mCurrentFrame.mvbOutlier[i]) {
                    pMP->IncreaseFound();
                    if (!mbOnlyTracking || pMP->Observations() > 0) {
                        mnMatchesInliers++;
                    }
                } else if (mSensor == System::STEREO) {
                    pMP = nullptr;
                }
            }
        }


        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
            return false;

        if (mnMatchesInliers < 30)
            return false;
        else
            return true;
    }




    bool Tracking::NeedNewKeyFrame() {
        if (mbOnlyTracking)
            return false;

        // If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
            return false;

        const int nKFs = mpMap->KeyFramesInMap();

        // Do not insert keyframes if not enough frames have passed from last relocalisation
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
            return false;

        // Tracked MapPoints in the reference keyframe
        int nMinObs = 3;
        if (nKFs <= 2)
            nMinObs = 2;
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

        // Local Mapping accept keyframes?
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

        // Check how many "close" points are being tracked and how many could be potentially created.
        int nNonTrackedClose = 0;
        int nTrackedClose = 0;
        if (mSensor != System::MONOCULAR) {
            for (int i = 0; i < mCurrentFrame.N; i++) {
                if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                        nTrackedClose++;
                    else
                        nNonTrackedClose++;
                }
            }
        }

        bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

        // Thresholds
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        if (mSensor == System::MONOCULAR)
            thRefRatio = 0.9f;

        // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
        // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
        //Condition 1c: tracking is weak
        const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
        // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15);

        if ((c1a || c1b || c1c) && c2) {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            if (bLocalMappingIdle) {
                return true;
            } else {
                mpLocalMapper->InterruptBA();
                if (mSensor != System::MONOCULAR) {
                    if (mpLocalMapper->KeyframesInQueue() < 3)
                        return true;
                    else
                        return false;
                } else
                    return false;
            }
        } else
            return false;
    }



    void Tracking::CreateNewKeyFrame() {
        if (!mpLocalMapper->SetNotStop(true))
            return;

        auto pKF = std::make_unique<KeyFrame>(mCurrentFrame, mpMap, mpKeyFrameDB);
        mpReferenceKF = pKF.get();
        mCurrentFrame.mpReferenceKF = pKF.get();

        if (mSensor != System::MONOCULAR) {
            mCurrentFrame.UpdatePoseMatrices();

            vector<pair<float, int>> vDepthIdx;
            vDepthIdx.reserve(mCurrentFrame.N);

            for (int i = 0; i < mCurrentFrame.N; i++) {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0) {
                    vDepthIdx.emplace_back(z, i);
                }
            }

            if (!vDepthIdx.empty()) {
                sort(vDepthIdx.begin(), vDepthIdx.end());

                int nPoints = 0;
                for (auto [depth, i] : vDepthIdx) {
                    bool bCreateNew = false;
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    if (!pMP || pMP->Observations() < 1) {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapPoints[i] = nullptr;
                    }

                    if (bCreateNew) {
                        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                        auto pNewMP = std::make_unique<MapPoint>(x3D, pKF.get(), mpMap);

                        pNewMP->AddObservation(pKF.get(), i);
                        pKF->AddMapPoint(pNewMP.get(), i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpMap->AddMapPoint(pNewMP.get());

                        mCurrentFrame.mvpMapPoints[i] = pNewMP.get();
                        pNewMP.release();
                        nPoints++;
                    } else {
                        nPoints++;
                    }

                    if (depth > mThDepth && nPoints > 100)
                        break;
                }
            }
        }

        mpLocalMapper->InsertKeyFrame(pKF.release());
        mpLocalMapper->SetNotStop(false);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKF.release();
    }

    void Tracking::SearchLocalPoints() {
        // Do not search map points already matched
        for (auto& pMP : mCurrentFrame.mvpMapPoints) {
            if (pMP) {
                if (pMP->isBad()) {
                    pMP = nullptr;
                } else {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

        // Project points in frame and check its visibility
        for (auto pMP : mvpLocalMapPoints) {
            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId || pMP->isBad()) {
                continue;
            }

            // Project (this fills MapPoint variables for matching)
            if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
                pMP->IncreaseVisible();
                ++nToMatch;
            }
        }

        if (nToMatch > 0) {
            ORBmatcher matcher(0.8);
            int th = (mSensor == System::RGBD) ? 3 : 1;

            // If the camera has been relocalised recently, perform a coarser search
            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;
            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
        }
    }

    void Tracking::UpdateLocalMap() {
        // This is for visualization
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        // Update
        UpdateLocalKeyFrames();
        UpdateLocalPoints();
    }

    void Tracking::UpdateLocalPoints() {
        mvpLocalMapPoints.clear();

        for (const auto pKF : mvpLocalKeyFrames) {
            const auto& vpMPs = pKF->GetMapPointMatches();

            for (const auto pMP : vpMPs) {
                if (!pMP || pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId) continue;
                if (!pMP->isBad()) {
                    mvpLocalMapPoints.emplace_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }


    void Tracking::UpdateLocalKeyFrames() {
        // Each map point vote for the keyframes in which it has been observed
        map<KeyFrame *, int> keyframeCounter;
        int N = mCurrentFrame.N;
        for (int i = 0; i < N; i++) {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (pMP && !pMP->isBad()) {
                const auto& observations = pMP->GetObservations();
                for (const auto& [keyFrame, _] : observations) {
                    keyframeCounter[keyFrame]++;
                }
            } else {
                mCurrentFrame.mvpMapPoints[i] = nullptr;
            }
        }


        if (keyframeCounter.empty())
            return;

        int max = 0;
        KeyFrame *pKFmax = nullptr;

        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

        // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for (const auto& [pKF, count] : keyframeCounter) {
            if (pKF->isBad()) continue;

            if (count > max) {
                max = count;
                pKFmax = pKF;
            }

            mvpLocalKeyFrames.emplace_back(pKF);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }



        // Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for (auto *pKF : mvpLocalKeyFrames) {

            // Limit the number of keyframes
            if (mvpLocalKeyFrames.size() > 80)
                break;


            for (auto *pNeighKF : pKF->GetBestCovisibilityKeyFrames(10)) {
                if (!pNeighKF->isBad() && pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.emplace_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }


            for (auto *pChildKF : pKF->GetChilds()) {
                if (!pChildKF->isBad() && pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.emplace_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }

            KeyFrame *pParent = pKF->GetParent();
            if (pParent && pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                mvpLocalKeyFrames.emplace_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }

        }

        if (pKFmax) {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    bool Tracking::Relocalization() {
        // Compute Bag of Words Vector
        mCurrentFrame.ComputeBoW();

        // Relocalization is performed when tracking is lost
        // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation

        auto vpCandidateKFs = mpMap->GetAllKeyFrames();

        if (vpCandidateKFs.empty()) {

            return false;
        }

        const int nKFs = vpCandidateKFs.size();


        // We perform first an ORB matching with each candidate
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75, true);

        vector<PnPsolver *> vpPnPsolvers(nKFs);
        vector<vector<MapPoint *>> vvpMapPointMatches(nKFs);
        vector<bool> vbDiscarded(nKFs);

        int nCandidates = 0;

        for (int i = 0; i < nKFs; i++) {
            auto *pKF = vpCandidateKFs[i];
            if (pKF->isBad() || matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]) < 15) {
                vbDiscarded[i] = true;
                continue;
            }
//            auto pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
            auto pSolver = make_unique<PnPsolver>(mCurrentFrame, vvpMapPointMatches[i]);

            pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
            vpPnPsolvers[i] = pSolver.get();
            pSolver.release();
            nCandidates++;


        }

        // Alternatively perform some iterations of P4P RANSAC
        // Until we found a camera pose supported by enough inliers
        bool bMatch = false;
        ORBmatcher matcher2(0.9, true);

        while (nCandidates > 0 && !bMatch) {
            for (int i = 0; i < nKFs; i++) {
                if (vbDiscarded[i])
                    continue;

                // Perform 5 Ransac Iterations
                auto* pSolver = vpPnPsolvers[i];
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

//                cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);
                auto Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // If Ransac reachs max. iterations discard keyframe
                if (bNoMore) {
                    vbDiscarded[i] = true;
                    nCandidates--;
                    continue; //////////////////////////////////////////////////
                }
                if (Tcw.empty()) continue;

                // If a Camera Pose is computed, optimize
                    Tcw.copyTo(mCurrentFrame.mTcw);

                    set<MapPoint *> sFound;

                    const int np = vbInliers.size();


                for (int j = 0; j < np; ++j) {
                    mCurrentFrame.mvpMapPoints[j] = vbInliers[j] ? vvpMapPointMatches[i][j] : nullptr;
                    if (vbInliers[j]) sFound.insert(vvpMapPointMatches[i][j]);
                }


                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    if (nGood < 10)
                        continue;

                    for (int io = 0; io < mCurrentFrame.N; io++)
                        if (mCurrentFrame.mvbOutlier[io])
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(nullptr);

                    // If few inliers, search by projection in a coarse window and optimize again
                    if (nGood < 50) {
                        int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10,
                                                                      100);

                        if (nadditional + nGood < 50) continue;

                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if (nGood > 30 && nGood < 50) {
                                sFound.clear();
                                int N = mCurrentFrame.N;
                                for (int ip = 0; ip < N; ip++)
                                    if (mCurrentFrame.mvpMapPoints[ip])
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3,
                                                                          64);

                                // Final optimization
                                if (nadditional + nGood < 50) continue;

                                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                    for (int io = 0; io < mCurrentFrame.N; io++)
                                        if (mCurrentFrame.mvbOutlier[io])
                                            mCurrentFrame.mvpMapPoints[io] = nullptr;

                            }

                    }


                    // If the pose is supported by enough inliers stop ransacs and continue
                    if (nGood >= 50) {
                        bMatch = true;
                        break;
                    }

            }
        }

        if (!bMatch) {

            return false;
        } else {
            mnLastRelocFrameId = mCurrentFrame.mnId;

            cv::Mat pose = mCurrentFrame.mTcw;
            cv::Mat R_wc = pose.rowRange(0, 3).colRange(0, 3);
            cv::Mat t_wc = pose.rowRange(0, 3).col(3);

            cv::Mat Rwc = pose.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat twc = -Rwc * pose.rowRange(0, 3).col(3);

            return true;
        }

    }

    void Tracking::Reset() {
        mpViewer->RequestStop();

        cout << "System Reseting" << endl;
        while (!mpViewer->isStopped())
            usleep(3000);

        // Reset Local Mapping
        cout << "Reseting Local Mapper...";
        mpLocalMapper->RequestReset();
        cout << " done" << endl;

        // Reset Loop Closing
        cout << "Reseting Loop Closing...";
        mpLoopClosing->RequestReset();
        cout << " done" << endl;

        // Clear BoW Database
        cout << "Reseting Database...";
        mpKeyFrameDB->clear();
        cout << " done" << endl;

        // Clear Map (this erase MapPoints and KeyFrames)
        mpMap->clear();

        KeyFrame::nNextId = 0;
        Frame::nNextId = 0;
        mState = NO_IMAGES_YET;

        if (mpInitializer) {
            delete mpInitializer;
            mpInitializer = nullptr;
        }

        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();

        mpViewer->Release();
    }

    void Tracking::ChangeCalibration(const string &strSettingPath) {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        Frame::mbInitialComputations = true;
    }

    void Tracking::InformOnlyTracking(const bool &flag) {
        mbOnlyTracking = flag;
    }


} //namespace ORB_SLAM


