#ifndef PNN_NOHF_HPP_
#define PNN_NOHF_HPP_

#include "FAST/ProcessObject.hpp"
#include "FAST/ExecutionDevice.hpp"
#include "FAST/Data/Image.hpp"

namespace fast {

class PnnNoHf : public ProcessObject {
    FAST_OBJECT(PnnNoHf)
    public:
        //void setMaskSize(unsigned char maskSize);
        //void setStandardDeviation(float stdDev);
        void setOutputType(DataType type);
        ~PnnNoHf();
    private:
        PnnNoHf();
        void execute();
        void waitToFinish();
        void initVolumeCube(Image::pointer input);
        std::vector<Image::pointer> getImagesAround(Vector3f worldPoint);
        void filterOutImagesByDfDom(std::vector<Image::pointer> neighbourFrames, int domDir, Vector3f worldPoint, float dfDom);
        void accumulateValue(Vector3i pointVoxelPos, float addValue, int channel);
        Vector3f getFramePointPosition(Image::pointer frame, int x, int y);
        void executeAlgorithmOnHost();
        //void createMask(Image::pointer input, uchar maskSize, bool useSeperableFilter);
        //void recompileOpenCLCode(Image::pointer input);

        //char mMaskSize;
        //float mStdDev;
        float dv; //resolution?
        float Rmax;
        bool volumeCalculated;
        bool volumeInitialized;
        Image::pointer firstFrame;
        Image::pointer output;
        Image::pointer outputImg;
        bool firstFrameNotSet;
        bool reachedEndOfStream;
        
        std::vector<Image::pointer> frameList;
        Image::pointer VoxelsValNWeight;
        float * VoxelValues;
        float * VoxelWeights;
        Vector3f zeroPoints;
        Vector3i volumeSize;

        ImageAccess::pointer volAccess;

        int iterartorCounter;


        cl::Buffer mCLMask;
        //float * mMask;
        //bool mRecreateMask;

        cl::Kernel mKernel;
        unsigned char mDimensionCLCodeCompiledFor;
        DataType mTypeCLCodeCompiledFor;
        DataType mOutputType;
        bool mOutputTypeSet;

};

} // end namespace fast

#endif /* PNN_NOHF_HPP_ */
