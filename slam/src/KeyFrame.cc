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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "RaspberryKeyFrame.h"
#include<mutex>

#include <boost/archive/text_oarchive.hpp>
// #include <boost/archive/binary_iarchive.hpp>


namespace boost{
    namespace serialization{
        
    }
}

namespace ORB_SLAM2
{

long unsigned int KeyFrame::nNextId=0;

KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap)
{

    image = F.image;
    
    mnId=nNextId++;

    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

    SetPose(F.mTcw);    
}

// Default serializing Constructor
KeyFrame::KeyFrame():
    mnFrameId(0),  mTimeStamp(0.0), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(0.0), mfGridElementHeightInv(0.0),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(0.0), fy(0.0), cx(0.0), cy(0.0), invfx(0.0), invfy(0.0),
    mbf(0.0), mb(0.0), mThDepth(0.0), N(0), mnScaleLevels(0), mfScaleFactor(0),
    mfLogScaleFactor(0.0),
    mnMinX(0), mnMinY(0), mnMaxX(0),
    mnMaxY(0)
{
#if 0
    mnId=nNextId++;

    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

    SetPose(F.mTcw);    
#endif
}

#ifndef _BAR_
template<class Archive>
    void KeyFrame::save(Archive & ar, const unsigned int version) const
    {
        ar & const_cast<cv::Mat &> (image);

        int nItems;bool is_id;bool has_parent = false;
        long unsigned int t_nId;
        //int ordered_weight;
        ar & nNextId;
        int ConKfWeight;
        
        //if (mbToBeErased)
            //return;
        //if (mbBad)
            //return;
        ar & const_cast<long unsigned int &> (mnId);
        ar & const_cast<long unsigned int &> (mnFrameId);
        ar & const_cast<double &> (mTimeStamp);
        ar & const_cast<int &> (mnGridCols);
        ar & const_cast<int &> (mnGridRows);
        ar & const_cast<float &>  (mfGridElementWidthInv);
        ar & const_cast<float &>  (mfGridElementHeightInv);
        ar & const_cast<long unsigned int &> (mnTrackReferenceForFrame);
        ar & const_cast<long unsigned int &> (mnFuseTargetForKF);
        ar & const_cast<long unsigned int &> (mnBALocalForKF);
        ar & const_cast<long unsigned int &> (mnBAFixedForKF);
        ar & const_cast<long unsigned int &> (mnLoopQuery);
        ar & const_cast<int &> (mnLoopWords);
        ar & const_cast<float &> (mLoopScore);
        ar & const_cast<long unsigned int &> (mnRelocQuery);
        ar & const_cast<int &> (mnRelocWords);
        ar & const_cast<float &> (mRelocScore);
        ar & const_cast<cv::Mat &> (mTcwGBA);
        ar & const_cast<cv::Mat &> (mTcwBefGBA);
        ar & const_cast<long unsigned int &> (mnBAGlobalForKF);
        ar & const_cast<float &> (fx);
        ar & const_cast<float &> (fy);
        ar & const_cast<float &> (cx);
        ar & const_cast<float &> (cy);
        ar & const_cast<float &> (invfx);
        ar & const_cast<float &> (invfy);
        ar & const_cast<float &> (mbf);
        ar & const_cast<float &> (mb);
        ar & const_cast<float &> (mThDepth);
        ar & const_cast<int &> (N);
        ar & const_cast<std::vector<cv::KeyPoint> &> (mvKeys);
        ar & const_cast<std::vector<cv::KeyPoint> &> (mvKeysUn);
        ar & const_cast<std::vector<float> &> (mvuRight);
        ar & const_cast<std::vector<float> &> (mvDepth);
        ar & const_cast<cv::Mat &> (mDescriptors);
        ar & const_cast<cv::Mat &> (mTcp);
        ar & const_cast<int &> (mnScaleLevels);
        ar & const_cast<float &> (mfScaleFactor);
        ar & const_cast<float &> (mfLogScaleFactor);
        ar & const_cast<std::vector<float> &> (mvScaleFactors);
        ar & const_cast<std::vector<float> &> (mvLevelSigma2);
        ar & const_cast<std::vector<float> &> (mvInvLevelSigma2);

        ar & const_cast<int &> (mnMinX);
        ar & const_cast<int &> (mnMinY);
        ar & const_cast<int &> (mnMaxX);
        ar & const_cast<int &> (mnMaxY);
        ar & const_cast<cv::Mat &> (mK);
        ar & const_cast<cv::Mat &> (Tcw);
        ar & const_cast<cv::Mat &> (Twc);
        ar & const_cast<cv::Mat &> (Ow);
        ar & const_cast<cv::Mat &> (Cw);
        // Save each map point id
        nItems = mvpMapPoints.size();
        ar & nItems;
        
        //cout << "{INFO}mvpMapPoints nItems -" << nItems << endl;
        for (std::vector<MapPoint*>::const_iterator it = mvpMapPoints.begin(); it != mvpMapPoints.end(); ++it) {        
            if (*it == NULL)
            {
                is_id = false;
                ar & is_id;
                continue;
            }
            else
            {
                is_id = true;
                ar & is_id;
                t_nId =  (**it).mnId;
                //cout << "[" << t_nId <<"]";

                ar & t_nId;

            }
 
            
        }

        // Grid
        ar & const_cast<std::vector< std::vector <std::vector<size_t> > > &> (mGrid);
         nItems = mConnectedKeyFrameWeights.size();
         ar & nItems;

         for (std::map<KeyFrame*,int>::const_iterator it = mConnectedKeyFrameWeights.begin(); 
                it != mConnectedKeyFrameWeights.end();
                ++it) 
         {        
            if (it->first == NULL)
            {
                is_id = false;
                ar & is_id;
                continue;
            }
            else
            {
                is_id = true;
                ar & is_id;
                t_nId =  it->first->mnId;
                ar & t_nId;
                ConKfWeight = it->second;
                ar & ConKfWeight;
            }
         }
         // Save each mvpOrderedConnectedKeyFrames
        nItems = mvpOrderedConnectedKeyFrames.size();
        ar & nItems;
        
        for (std::vector<KeyFrame*>::const_iterator it = mvpOrderedConnectedKeyFrames.begin(); 
                it != mvpOrderedConnectedKeyFrames.end(); 
                ++it) {        
            if (*it == NULL)
            {
                is_id = false;
                ar & is_id;
                continue;
            }
            else
            {
                is_id = true;
                ar & is_id;
                t_nId =  (**it).mnId;
                ar & t_nId;
            }            
        }
        // Save Each mvOrderedWeights
#if 0
        
        nItems = mvOrderedWeights.size();
        
        ar & nItems;
        
        for (std::vector<int>::const_iterator it = mvOrderedWeights.begin(); 
                it != mvOrderedWeights.end(); 
                ++it) 
        {        
            ordered_weight =  (*it);
            ar & ordered_weight;
                    
        }
#endif
        ar &  const_cast<std::vector<int> &>(mvOrderedWeights);

        // Spanning Tree
        ar & const_cast<bool &> (mbFirstConnection);
        
        if (mpParent)
        {
            has_parent = true;
            ar & has_parent;
            ar & mpParent->mnId;
        }
        else
        {
            has_parent = false;
            ar & has_parent;
            //ar & mpParent->mnId;
        }
        // Save each child Frame id
        nItems = mspChildrens.size();
        ar & nItems;
            for (std::set<KeyFrame*>::const_iterator it = mspChildrens.begin(); it != mspChildrens.end(); ++it) {        
            if (*it == NULL)
            {
                is_id = false;
                ar & is_id;
                continue;
            }
            else
            {
                is_id = true;
                ar & is_id;
                t_nId =  (**it).mnId;
                //cout << "[" << t_nId <<"]";
                ar & t_nId;
            } 
            
        }
        // Save each Loop Edge id
        nItems = mspLoopEdges.size();
        ar & nItems;
            for (std::set<KeyFrame*>::const_iterator it = mspLoopEdges.begin(); it != mspLoopEdges.end(); ++it) {        
            if (*it == NULL)
            {
                is_id = false;
                ar & is_id;
                continue;
            }
            else
            {
                is_id = true;
                ar & is_id;
                t_nId =  (**it).mnId;
                //cout << "[" << t_nId <<"]";
                ar & t_nId;
            } 
            
        }

        ar & const_cast<bool &> (mbNotErase);
        ar & const_cast<bool &> (mbToBeErased);
        ar & const_cast<bool &> (mbBad);
        ar & const_cast<float &> (mHalfBaseline);
        // cout << "{INFO}mvpMapPoints nItems fin"<< endl;
        //cout << "Save Map :  KF stat is : " << endl;
        //t_nId = has_parent?mpParent->mnId:0;
        //cout << "KF mnId = " << mnId << " Parent ID " <<t_nId <<" mspLoopEdges.size() = " << nItems <<endl;

       
    }

    template<class Archive>
    void KeyFrame::load(Archive & ar, const unsigned int version)
    {
        ar & const_cast<cv::Mat &> (image);

        id_map storer;
        int nItems;bool is_id = false;
        bool has_parent = false;
        long unsigned int t_nId;
        ar & nNextId;
        ar & const_cast<long unsigned int &> (mnId);
        int ConKfWeight = 0;
        //cout << "{INFO}Keyframe Load - " << mnId << endl;

        ar & const_cast<long unsigned int &> (mnFrameId);
        ar & const_cast<double &> (mTimeStamp);
        ar & const_cast<int &> (mnGridCols);
        ar & const_cast<int &> (mnGridRows);
        ar & const_cast<float &>  (mfGridElementWidthInv);
        ar & const_cast<float &>  (mfGridElementHeightInv);
        ar & const_cast<long unsigned int &> (mnTrackReferenceForFrame);
        ar & const_cast<long unsigned int &> (mnFuseTargetForKF);
        ar & const_cast<long unsigned int &> (mnBALocalForKF);
        ar & const_cast<long unsigned int &> (mnBAFixedForKF);
        ar & const_cast<long unsigned int &> (mnLoopQuery);
        ar & const_cast<int &> (mnLoopWords);
        ar & const_cast<float &> (mLoopScore);
        ar & const_cast<long unsigned int &> (mnRelocQuery);
        ar & const_cast<int &> (mnRelocWords);
        ar & const_cast<float &> (mRelocScore);
        ar & const_cast<cv::Mat &> (mTcwGBA);
        ar & const_cast<cv::Mat &> (mTcwBefGBA);
        ar & const_cast<long unsigned int &> (mnBAGlobalForKF);
        ar & const_cast<float &> (fx);
        ar & const_cast<float &> (fy);
        ar & const_cast<float &> (cx);
        ar & const_cast<float &> (cy);
        ar & const_cast<float &> (invfx);
        ar & const_cast<float &> (invfy);
        ar & const_cast<float &> (mbf);
        ar & const_cast<float &> (mb);
        ar & const_cast<float &> (mThDepth);
        ar & const_cast<int &> (N);
        ar & const_cast<std::vector<cv::KeyPoint> &> (mvKeys);
        ar & const_cast<std::vector<cv::KeyPoint> &> (mvKeysUn);
        ar & const_cast<std::vector<float> &> (mvuRight);
        ar & const_cast<std::vector<float> &> (mvDepth);
        ar & const_cast<cv::Mat &> (mDescriptors);
        ar & const_cast<cv::Mat &> (mTcp);
        ar & const_cast<int &> (mnScaleLevels);
        ar & const_cast<float &> (mfScaleFactor);
        ar & const_cast<float &> (mfLogScaleFactor);
        ar & const_cast<std::vector<float> &> (mvScaleFactors);
        ar & const_cast<std::vector<float> &> (mvLevelSigma2);
        ar & const_cast<std::vector<float> &> (mvInvLevelSigma2);

        ar & const_cast<int &> (mnMinX);
        ar & const_cast<int &> (mnMinY);
        ar & const_cast<int &> (mnMaxX);
        ar & const_cast<int &> (mnMaxY);
        ar & const_cast<cv::Mat &> (mK);
        ar & const_cast<cv::Mat &> (Tcw);
        ar & const_cast<cv::Mat &> (Twc);
        ar & const_cast<cv::Mat &> (Ow);
        ar & const_cast<cv::Mat &> (Cw);
        // Load each map point id
        ar & nItems;
        mvpMapPoints.resize(nItems);
        //mmMapPoints_nId.resize(nItems);
        int j=0;
        for (int i = 0; i < nItems; ++i) { 

            ar & is_id;
            if (is_id)
            {
                j++;
                ar & t_nId;
                storer.is_valid = true;
                storer.id= t_nId;
                mmMapPoints_nId[i] = storer;
            }
            else
            {
                storer.is_valid = false;
                storer.id= 0;
                mmMapPoints_nId[i] = storer;
            }
        }
        
        //cout << "KF " << mnId <<" valid points = " << j << "invalid points = " << (nItems - j) << endl;
        // Grid
        ar & const_cast<std::vector< std::vector <std::vector<size_t> > > &> (mGrid);

        ar & nItems;
        //mConnectedKeyFrameWeights_nId.resize(nItems);
       //mConnectedKeyFrameWeights.resize(nItems);
        for (int i = 0; i < nItems; ++i) { 

            ar & is_id;
            if (is_id)
            {
                ar & t_nId;
                ar & ConKfWeight;
                mConnectedKeyFrameWeights_nId[t_nId] = ConKfWeight;
            }
            else
            {

            }
        }

         // Load each mvpOrderedConnectedKeyFrames id
        ar & nItems;
        j = 0;
        //mvpOrderedConnectedKeyFrames.resize(nItems);
        //mvpOrderedConnectedKeyFrames_nId.resize(nItems);
        for (int i = 0; i < nItems; ++i) {
            ar & is_id;
            if (is_id)
            {
                j++;
                ar & t_nId;
                storer.is_valid = true;
                storer.id= t_nId;
                mvpOrderedConnectedKeyFrames_nId[i] = storer;
            }
            else
            {
                storer.is_valid = false;
                storer.id= 0;
                mvpOrderedConnectedKeyFrames_nId[i] = storer;
            }
        }
        // Load each mvOrderedWeights
        ar & const_cast<std::vector<int> &>(mvOrderedWeights);

        ar & const_cast<bool &> (mbFirstConnection);
        
        // Spanning Tree    
        ar & has_parent;
        if (has_parent)
        {
            mparent_KfId_map.is_valid = true;
            ar & mparent_KfId_map.id;      
        }
        else
        {
            mparent_KfId_map.is_valid = false; 
            mparent_KfId_map.id = 0;
        }
        // load each child Frame id
        ar & nItems;
        //mspChildrens.resize(nItems);
        //mmChildrens_nId.resize(nItems);

       for (int i = 0; i < nItems; ++i) { 

            ar & is_id;
            if (is_id)
            {
                ar & t_nId;
                storer.is_valid = true;
                storer.id= t_nId;
                mmChildrens_nId[i] = storer;
            }
            else
            {
                storer.is_valid = false;
                storer.id= 0;
                mmChildrens_nId[i] = storer;
            }
        }

        // Load each Loop Edge Frame id
        ar & nItems;
        //mspLoopEdges.resize(nItems);
        //mmLoopEdges_nId.resize(nItems);

       for (int i = 0; i < nItems; ++i) { 

            ar & is_id;
            if (is_id)
            {
                ar & t_nId;
                storer.is_valid = true;
                storer.id= t_nId;
                mmLoopEdges_nId[i] = storer;
            }
            else
            {
                storer.is_valid = false;
                storer.id= 0;
                mmLoopEdges_nId[i] = storer;
            }
        }

        ar & const_cast<bool &> (mbNotErase);
        ar & const_cast<bool &> (mbToBeErased);
        ar & const_cast<bool &> (mbBad);
        ar & const_cast<float &> (mHalfBaseline);
        //cout << "Load Map :  KF stat is : " << endl;
        //cout << "KF mnId = " << mnId << " Parent ID " <<mparent_KfId_map.id <<" mspLoopEdges.size() = " << nItems <<endl;

    }


// Explicit template instantiation
template void KeyFrame::save<boost::archive::binary_oarchive>(
    boost::archive::binary_oarchive &, 
    const unsigned int) const;
template void KeyFrame::save<boost::archive::binary_iarchive>(
    boost::archive::binary_iarchive &, 
    const unsigned int) const;
template void KeyFrame::load<boost::archive::binary_oarchive>(
    boost::archive::binary_oarchive &, 
    const unsigned int);
template void KeyFrame::load<boost::archive::binary_iarchive>(
    boost::archive::binary_iarchive &, 
    const unsigned int);
#endif


void KeyFrame::SetMapPoints(std::vector<MapPoint*> spMapPoints)
    {

        std::unordered_map<long unsigned int, MapPoint*> mapPointLookup;
        for (auto* mapPoint : spMapPoints)
        {
            mapPointLookup[mapPoint->mnId] = mapPoint;
        }

        int j = 0;
        for (const auto& [nid, mapData] : mmMapPoints_nId)
        {
            if (!mapData.is_valid)
            {
                mvpMapPoints[j++] = nullptr;
                continue;
            }

            auto iter = mapPointLookup.find(mapData.id);
            if (iter != mapPointLookup.end())
            {
                mvpMapPoints[j++] = iter->second;
            }
            else
            {
                // Map point [mapData.id] not found in KF
                mvpMapPoints[j++] = nullptr;
            }
        }
    }


void KeyFrame::SetSpanningTree(std::vector<KeyFrame*> vpKeyFrames)
{

    std::unordered_map<long unsigned int, KeyFrame*> keyFrameLookup;
    for (auto* kf : vpKeyFrames) {
        keyFrameLookup[kf->mnId] = kf;
    }

    // Search Parent
    if (mparent_KfId_map.is_valid) {
        auto it = keyFrameLookup.find(mparent_KfId_map.id);
        if (it != keyFrameLookup.end()) {
            mpParent = it->second;
        } else {
            std::cout << "\nParent KF [" << mparent_KfId_map.id << "] not found for KF " << mnId << std::endl;
            mpParent = nullptr;
        }
    }

    // Search Children
    for (const auto& [nid, childData] : mmChildrens_nId) {
        if (childData.is_valid) {
            auto it = keyFrameLookup.find(childData.id);
            if (it != keyFrameLookup.end()) {
                mspChildrens.insert(it->second);
            } else {
                std::cout << "\nChild [" << childData.id << "] not found for KF " << mnId << std::endl;
            }
        }
    }

    // Search Loop Edges
    for (const auto& [nid, loopData] : mmLoopEdges_nId) {
        if (loopData.is_valid) {
            auto it = keyFrameLookup.find(loopData.id);
            if (it != keyFrameLookup.end()) {
                mspLoopEdges.insert(it->second);
            } else {
                std::cout << "\nLoop Edge [" << loopData.id << "] not found for KF " << mnId << std::endl;
            }
        }
    }


}





void KeyFrame::SetGridParams(std::vector<KeyFrame*> vpKeyFrames)
{
    long unsigned int id; int weight;  
    bool Kf_found = false;  
    //cout << "KF" << mnId <<" valid indexes-" << endl;
    int j = 0; 
    int ctr = 0;
    bool is_valid = false;

    
    // Set up mConnectedKeyFrameWeights
    for (map<long unsigned int, int>::iterator it = mConnectedKeyFrameWeights_nId.begin(); 
            it != mConnectedKeyFrameWeights_nId.end(); 
            j++,++it) 
    {
        id = it->first;
        weight = it->second;        
        {
            
            for(std::vector<KeyFrame*>::iterator mit=vpKeyFrames.begin(); mit !=vpKeyFrames.end(); mit++)
            {
                KeyFrame* pKf = *mit;
               
                if(id == pKf->mnId)
                {   
                    mConnectedKeyFrameWeights[pKf] = weight;
                    break;
                }
            }
            
        }
    }

    // Set up mvpOrderedConnectedKeyFrames
    j = 0;
    for (std::map<long unsigned int,id_map>::iterator it = mvpOrderedConnectedKeyFrames_nId.begin(); 
            it != mvpOrderedConnectedKeyFrames_nId.end(); 
            ++it) 
    {
        is_valid = it->second.is_valid; 
        if (!is_valid)  
        {   
            continue; 
            ;//mvpOrderedConnectedKeyFrames[j] = NULL;
        }
        else
        {
            id = it->second.id;  
            
            Kf_found = false;
            for(std::vector<KeyFrame*>::iterator mit=vpKeyFrames.begin(); mit !=vpKeyFrames.end(); mit++)
            {
                KeyFrame* pKf = *mit;

                if(id == pKf->mnId)
                {    
                    ctr ++;
                    mvpOrderedConnectedKeyFrames.push_back(pKf);
                    Kf_found = true;
                    break;
                }
            }
            if (Kf_found == false)
            {
                //cout << "[" << id <<"] not found in KF " << mnId << endl;
                
            }

        }

    }
}

void KeyFrame::SetMap(Map* map)
{
    mpMap = map;   
}

void KeyFrame::SetKeyFrameDatabase(KeyFrameDatabase* pKeyFrameDB)
{
    mpKeyFrameDB = pKeyFrameDB;
}

void KeyFrame::SetORBvocabulary(ORBVocabulary* pORBvocabulary)
{
    mpORBvocabulary = pORBvocabulary;
}

void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw;

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    Cw = Twc*center;
}

cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}

cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for (const auto& [kf, weight] : mConnectedKeyFrameWeights) {
        vPairs.emplace_back(weight, kf);
    }

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for (const auto& [weight, kf] : vPairs) {
        lKFs.push_front(kf);
        lWs.push_front(weight);
    }

    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for (const auto& [kf, weight] : mConnectedKeyFrameWeights) {
        s.insert(kf);
    }
    return s;
}

vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    return (mvpOrderedConnectedKeyFrames.size() < N) ? mvpOrderedConnectedKeyFrames : vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);

}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    if(it==mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}


void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(auto pMP : mvpMapPoints)
    {
        if(pMP && !pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(const auto& pMP : mvpMapPoints)
    {
        if(pMP && !pMP->isBad() && (!bCheckObs || pMP->Observations() >= minObs))
        {
            nPoints++;
        }
    }

    return nPoints;
}

vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

void KeyFrame::UpdateConnections()
{
    map<KeyFrame*,int> KFcounter;

    vector<MapPoint*> vpMP;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for(const auto& pMP : vpMP)
    {
        if(!pMP || pMP->isBad())
            continue;

        const auto& observations = pMP->GetObservations();

        for(const auto& [kf, _] : observations)
        {
            if(kf->mnId == mnId)
                continue;
            KFcounter[kf]++;
        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax= nullptr;
    int th = 15;

    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(auto [kf, count] : KFcounter)
    {
        if(count > nmax)
        {
            nmax = count;
            pKFmax = kf;
        }
        if(count >= th)
        {
            vPairs.emplace_back(count, kf);
            kf->AddConnection(this, count);
        }
    }

    if(vPairs.empty())
    {
        vPairs.emplace_back(nmax,pKFmax);
        pKFmax->AddConnection(this,nmax);
    }

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(const auto& [weight, kf] : vPairs)
    {
        lKFs.push_front(kf);
        lWs.push_front(weight);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        if(mbFirstConnection && mnId!=0)
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{   
  
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for(const auto& [kf, _] : mConnectedKeyFrameWeights)
        kf->EraseConnection(this);

    for(auto& mp : mvpMapPoints)
        if(mp)
            mp->EraseObservation(this);
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        if(mpParent)
            sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            for(KeyFrame* pKF : mspChildrens)
            {

                if(!pKF || pKF->isBad())
                    continue;
                

                // Check if a parent candidate is connected to the keyframe
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();


                for(const auto& connectedKF : vpConnected)
                {


                    for(const auto& parentCandidate : sParentCandidates)
                    {


                        if(connectedKF->mnId == parentCandidate->mnId)
                        {
                            

                            int w = pKF->GetWeight(connectedKF);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = connectedKF;
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                

                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if(!mspChildrens.empty() && mpParent)
            for(KeyFrame* child : mspChildrens)
                child->ChangeParent(mpParent);

        if (mpParent)
        {
            mpParent->EraseChild(this);
        
            mTcp = Tcw*mpParent->GetPoseInverse();
        }
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const auto &vCell = mGrid[ix][iy];
            for(const auto& idx : vCell)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[idx];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(idx);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}

float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2,3);
    for(const auto& pMP : vpMapPoints)
    {
        if(pMP)
        {
            const cv::Mat x3Dw = pMP->GetWorldPos();
            vDepths.emplace_back(Rcw2.dot(x3Dw) + zcw);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

void KeyFrame::rpi_save(const std::string& file_name)
{
    RaspberryKeyFrame* rpi_kf = new RaspberryKeyFrame(*this);
    {
    std::ofstream os(file_name);
    boost::archive::text_oarchive oa(os);
    //oa << mpKeyFrameDatabase;
    oa << rpi_kf;
    }

    delete rpi_kf;
}

void KeyFrame::align(const cv::Mat& R_align, const cv::Mat& mu_align)
{
    auto pose = GetPose();
    cv::Mat Rwc = pose.rowRange(0, 3).colRange(0, 3).clone().t();
    cv::Mat tcw = pose.rowRange(0, 3).col(3).clone();

    cv::Mat twc = -Rwc*tcw;

    cv::Mat aligned_Rwc = R_align * Rwc;
    cv::Mat aligned_twc = R_align * (twc - mu_align);

    cv::Mat align_pose;

    align_pose = cv::Mat::eye(4,4,CV_32F);

    cv::Mat aligned_Rcw = aligned_Rwc.t();
    cv::Mat aligned_tcw = -aligned_Rcw * aligned_twc;
    aligned_Rcw.copyTo(align_pose.rowRange(0,3).colRange(0,3));
    aligned_tcw.copyTo(align_pose.rowRange(0,3).col(3));

    SetPose(align_pose);
}

} //namespace ORB_SLAM
