#include "FAST/Algorithms/UsReconstruction/PnnNoHf.hpp"
#include "FAST/Exception.hpp"
#include "FAST/DeviceManager.hpp"
#include "FAST/Data/Image.hpp"
using namespace fast;

void PnnNoHf::setOutputType(DataType type){
    mOutputType = type;
    mOutputTypeSet = true;
    mIsModified = true;
}

PnnNoHf::PnnNoHf(){
    createInputPort<Image>(0);
    createOutputPort<Image>(0, OUTPUT_DEPENDS_ON_INPUT, 0);
    //create openCL prog here
    //--//createOpenCLProgram(std::string(FAST_SOURCE_DIR) + "Algorithms/GaussianSmoothingFilter/GaussianSmoothingFilter2D.cl", "2D");
    //--// store different compiled for settings (dimension/variables...)
    mIsModified = true; // needed?
    mOutputTypeSet = false;

    //volume;
    dv = 1.0;
    Rmax = 3.0; //2?
    volumeInitialized = false;
    firstFrameNotSet = true;
    frameList = {};
    //frameList.capacity = 1000;
}

PnnNoHf::~PnnNoHf(){
    //delete something
}

//void PnnNoHf::recompileOpenCLCode(Image::pointer input) {

Vector3f getImagePlaneNormal(Image::pointer frame){
    AffineTransformation::pointer imageTransformation = SceneGraph::getAffineTransformationFromData(frame);
    Vector3f p0 = imageTransformation->multiply(Vector3f(0, 0, 0));
    Vector3f p1 = imageTransformation->multiply(Vector3f(1, 0, 0));
    Vector3f p2 = imageTransformation->multiply(Vector3f(0, 1, 0));
    Vector3f imagePlaneNormal = (p1 - p0).cross(p2 - p0);
    imagePlaneNormal.normalize();
    return imagePlaneNormal;
}

Vector3i getImageComponentVector(Image::pointer frame, int x, int y){
    AffineTransformation::pointer imageTransformation = SceneGraph::getAffineTransformationFromData(frame);
    Vector3f x0 = imageTransformation->multiply(Vector3f(0, 0, 0));
    Vector3f x1 = imageTransformation->multiply(Vector3f(x, y, 0));
    Vector3f comp = (x1 - x0);
    Vector3i offset = Vector3i(comp(0), comp(1), comp(2));
    return offset;
    //TODO fix  Vector3f(1,1,0)-Vector3f(0,0,0) multiplies?
}

Vector3f PnnNoHf::getFramePointPosition(Image::pointer frame, int x, int y){
    AffineTransformation::pointer imageTransformation = SceneGraph::getAffineTransformationFromData(frame);
    Vector3f worldPos = imageTransformation->multiply(Vector3f(x, y, 0));
    worldPos -= zeroPoints;
    if (worldPos(0) > volumeSize(0) || worldPos(1) > volumeSize(1) || worldPos(2) > volumeSize(2)){
        return Vector3f(0, 0, 0); //Occures sometimes! Should not! Increase size of volume? Or have to change something?
    }
    return worldPos; //TODO relate to zeroPoint
}

Vector3i getRoundedIntVector3f(Vector3f v){
    return Vector3i(round(v(0)), round(v(1)), round(v(2)));
}

/*
Fetches images close to a point in the world coordinate system
* worldPoint: Point in world space
# Return list of Image-pointers for relevant Image frames (has to include all relevant frames)
*/
std::vector<Image::pointer> PnnNoHf::getImagesAround(Vector3f worldPoint){
    /*
    std::vector<Image::pointer> framesToCheck = {};
                if (last.isValid()){
                    framesToCheck.push_back(last);
                }
                if (next.isValid()){
                    framesToCheck.push_back(next);
                }
                framesToCheck.push_back(frame);
    */
    std::vector<Image::pointer> outputList = {};
    //TODO actual implementation
    return outputList;
}

/*
Filters out images to only include those within the appropriate dfDom distance
* neighbourFrames: List of Image-pointers for relevant Image frames (has to include all relevant frames)
* domDir: the dominant direction of imagePlaneNormal (0:x, 1:y, 2:z) Direction to iterate over
* worldPoint: Point in world space
* dfDom: Distance to iterate over domDir in each direction
# Return list of Image-pointers for relevant Image frames (has to include all relevant frames; but minimal number of images)
*/
void PnnNoHf::filterOutImagesByDfDom(std::vector<Image::pointer> neighbourFrames, int domDir, Vector3f worldPoint, float dfDom){
    //TODO use neighbourFrames and remove frames outside of dfDom distance

    //TODO make sure input "neighbourFrames" is affected outside afterwards, or change to returning that
}

/*
Get distance from point worldPoint to plane neighFrame along the imagePlaneNormal
* worldPoint: Point in world space
* imagePlaneNormal: Normal vector from worldPoint
* neighFrame: Image with transformed corners to be used for generating plane //TODO precalc planes for reuse?
# Return (interpolated) pixelvalue
*/
float getNormalDistance(Vector3f worldPoint, Vector3f imagePlaneNormal, Image::pointer neighFrame){
    float distance = 0.0;
    //TODO PRIO implement needs just math
    return distance;
}

/* 
Sample from neighFrame the point at pointed at by worldPoint and imagePlaneNormal
* worldPoint: Point in world space
* imagePlaneNormal: Normal vector from worldPoint
* neighFrame: Image with metadata(worldspace, corners etc) and pixelvalues to be sampled
# Return (interpolated) pixelvalue
*/
float sampleAtNormal(Vector3f worldPoint, Vector3f imagePlaneNormal, Image::pointer neighFrame){
    float pixelValue = 0.0;
    //TODO implement sampling

    return pixelValue;
}

void PnnNoHf::accumulateValue(Vector3i pointVoxelPos, float addValue, int channel){
    /*
    float oldP = volAccess->getScalar(pointVoxelPos, 0);
    float oldW = volAccess->getScalar(pointVoxelPos, 1);
    if (oldP < 0.0){ oldP = 0.0; } 
    if (oldW < 0.0){ oldW = 0.0; }
    float newP = oldP + p;
    float newW = oldW + w;
    volAccess->setScalar(pointVoxelPos, newP, 0);
    volAccess->setScalar(pointVoxelPos, newW, 1);
    */
    float oldValue = volAccess->getScalar(pointVoxelPos, channel);
    //if (oldValue < 0.0){ oldValue = 0.0; } // Hacky workaround from uninitialized cells
    float newValue = oldValue + addValue;
    volAccess->setScalar(pointVoxelPos, newValue, channel);
}

// CPU algoritme
//template <class T>
void PnnNoHf::executeAlgorithmOnHost(){//Image::pointer VoxelsValNWeight, Image::pointer output, std::vector<Image::pointer> frameList, float dv, float Rmax){
    //VoxelsValNWeight = Image::pointer::New();
    std::cout << "Started executing on host!" << std::endl;
    volAccess = VoxelsValNWeight->getImageAccess(accessType::ACCESS_READ_WRITE);
    /*
    for (int x = 0; x < 5; x++){
        for (int y = 0; y < 5; y++){
            for (int z = 0; z < 5; z++){
                Vector3i position = Vector3i(x, y, z);
                float voxelValue0 = volAccess->getScalar(position, 0);
                float voxelValue1 = volAccess->getScalar(position, 1);
            }
        }
    }*/
    //Image::pointer lastFrame = none;
    for (int i = 0; i < frameList.size(); i++){
        Image::pointer frame = frameList[i];
        //zeroPoints
        // # Finn dominerende rettning #
        Vector3f imagePlaneNormal = getImagePlaneNormal(frame);
        //Vector3f pointWorldPos = getFramePointPosition(frame, 40, 50);
        //Vector3i pointVoxelPos = getRoundedIntVector3f(pointWorldPos);
        float domVal = fabs(imagePlaneNormal(0)); //TODO do we have to take ABS of value here? Does the + vs - direction matter?
        int domDir = 0;
        if (fabs(imagePlaneNormal(1)) > domVal){
            domVal = fabs(imagePlaneNormal(1));
            domDir = 1;
        }
        if (fabs(imagePlaneNormal(2)) > domVal) {
            domVal = fabs(imagePlaneNormal(2));
            domDir = 2; 
        }
        // TODO adjust for or compare to output volume direction

        // # Go through output volume #
        ImageAccess::pointer frameAccess = frame->getImageAccess(accessType::ACCESS_READ);
        Image::pointer last;
        Image::pointer next;
        if (i != 0) { 
            last = frameList[i - 1]; 
        }
        //if (!frameList.empty()){ Image::pointer next = frameList.back(); }
        if (i < frameList.size()-1 ) { 
            next = frameList[i + 1]; 
        }

        for (int x = 0; x < frame->getWidth(); x++){
            for (int y = 0; y < frame->getHeight(); y++){
                // ## Beregn world basert voxelposisjon til pixelen ##
                Vector3f pointWorldPos = getFramePointPosition(frame, x, y); //Floating-point accuracy position
                Vector3i pointVoxelPos = getRoundedIntVector3f(pointWorldPos); //Discrete integerbased position

                // ## Find distance to last and next
                float d1 = 1.2; float d2 = 1.4;

                // ## Calculate df & dfz
                float maxNeighDist = std::max(d1, d2);
                float df = std::min(std::max(maxNeighDist, dv), Rmax);
                float dfz = df / domVal; // TODO verify that deviding by domVal is correct (or 1/domval?) // Is domVal in range under 1 or over 1?

                // ## Within dfz in each direction, find frames within this zone:
                float zeroDom = pointWorldPos(domDir);
                // Images = getImagesAround(pointWorldPos)
                std::vector<Image::pointer> neighbourFrames = getImagesAround(pointWorldPos); //TODO eller skal vi bruke pointVoxelPos her og nedenfor?
                // Images = filterOutImages(Images, domDir, zeroDom, dfz) #//float minDom = zeroDom - dfz; //float maxDom = zeroDom + dfz;
                filterOutImagesByDfDom(neighbourFrames, domDir, pointWorldPos, dfz);
                // TODO Store Images to reuse for next iteration (ie.use all accepted, or have a base with all frames close to this one based on 4 corners)
                // For Img in Images:
                for (Image::pointer neighFrame : neighbourFrames){
                    // ? Find riktig sample/beam i rawdata ?? For no bruk pointWorldPos/pointVoxelPos
                    // float d = getDistanceAtNormal(Point pointWorldPos, NormalVector imageNormal, Plane Img)
                    float dist = getNormalDistance(pointWorldPos, imagePlaneNormal, neighFrame);
                    // If ( d <= dfz ):
                    if (dist <= dfz){
                        /*ImageAccess::pointer nFrameAccess = neighFrame->getImageAccess(accessType::ACCESS_READ);
                        float p = nFrameAccess->getScalar((x, y), 0); //TODO actually find closest
                        float w = 1; // w = 1-d/df*/
                        // float p = sampleAtNormal(Point pointWorldPos, NormalVector imageNormal, Plane Img)
                        float p = sampleAtNormal(pointWorldPos, imagePlaneNormal, neighFrame);
                        // float w = 1 - d/df
                        float w = 1 - dist / df;
                        // accumulate p's and w's
                        accumulateValue(pointVoxelPos, p, 0);
                        accumulateValue(pointVoxelPos, p, 1);
                        //TODO further add 3rd component for time?
                    }
                }
                    
                
            }
        }
        /*
        Vector3f baselocation = frame->getTransformedBoundingBox().getCorners().row(0); //corner 0 local values (0,0,0)
        Vector3f adjustedBaseLocation = baselocation - zeroPoints;
        int sizeA, sizeB;
        if (domDir == 2){
            sizeA = frame->getWidth();
            sizeB = frame->getHeight();
        }
        else if (domDir == 1){
            sizeA = frame->getWidth();
            sizeB = frame->getDepth();
        }
        else{
            sizeA = frame->getHeight();
            sizeB = frame->getDepth();
        }
        for (int a = 0; a < sizeA; a++){ //TODO use domDir to decide if its actually x/y/z we are gonna leave out of iteration. Ie. call these a/b and asign them to the appropriate x/y/z + x/y/z
            for (int b = 0; b < sizeB; b++){
                //Index for 2 non-superior directions of imagePlaneNormal
                int x, y, z = 0;
                if (domDir == 2){
                    x = a; y = b;   
                } else if (domDir == 1){  
                    x = a; z = b;   
                } else {                   
                    y = a; z = b;   
                }
                //Find thickness according to last and next frame along the imagePlaneNormal
                Vector3i location = Vector3i(x, y, z); // TODO make sure x, y is transformed -> x/y need to be transformed now to reflect change in new coordinate system
                location(0) += round(adjustedBaseLocation(0)); //use round instead? int() does cutoff 158.94 -> 158
                location(1) += round(adjustedBaseLocation(1));
                location(2) += round(adjustedBaseLocation(2)); //TODO Location must be decided by unit vector in each direction times the number x/y/z
                Vector3ui volSize = VoxelsValNWeight->getSize();
                if (location(0) >= VoxelsValNWeight->getWidth()){
                    continue;
                }
                if (location(1) >= VoxelsValNWeight->getHeight()){
                    continue;
                }
                if (location(2) >= VoxelsValNWeight->getDepth()){
                    continue;
                }
                //todo: find dist to LAST and NEXT
                float d1 = 1.2; float d2 = 1.4;
                //Calculate
                float maxNeighDist = std::max(d1, d2);
                float df = std::min(std::max(maxNeighDist, dv), Rmax);
                float dfz = df / domVal; // TODO verify that deviding by domVal is correct (or 1/domval?)

                //Find pixels in frames within thickness (2x dfz) in Z-direction
                // for now just checking next and last TODO fix // These can also be limited outside the x/y for loop to show only frames within a certain distance from this frame (for instance using Rmax)
                std::vector<Image::pointer> framesToCheck = {};
                if (last.isValid()){
                    framesToCheck.push_back(last);
                }
                if (next.isValid()){
                    framesToCheck.push_back(next);
                }
                framesToCheck.push_back(frame);
                
                for (Image::pointer nFrame : framesToCheck){
                    // Given that frame has close enough pixels: Find pixel closest to point
                    ImageAccess::pointer nFrameAccess = nFrame->getImageAccess(accessType::ACCESS_READ);
                    float p = nFrameAccess->getScalar((x, y), 0); //TODO actually find closest
                    float w = 1; // w = 1-d/df
                    // Accumulate
                    float oldP = volAccess->getScalar(location, 0);
                    float oldW = volAccess->getScalar(location, 1);
                    if (oldP < 0.0){ oldP = 0.0; } // Hacky workaround from -51.00 cells
                    if (oldW < 0.0){ oldW = 0.0; }
                    float newP = oldP + p;
                    float newW = oldW + w;
                    volAccess->setScalar(location, newP, 0);
                    volAccess->setScalar(location, newW, 1);
                    //TODO further add 3rd component for time?
                }
                
            }
        }
        */
    }
    
    /*while (!frameList.empty()){
        Image::pointer frame = frameList.back();
        frameList.pop_back();
    }*/
    std::cout << "Calculating reconstructed volume!" << std::endl;
    // Finally, calculate reconstructed volume
    output = getStaticOutputData<Image>(0);
    output->create(VoxelsValNWeight->getSize(), VoxelsValNWeight->getDataType(), 1);
    ImageAccess::pointer outAccess = output->getImageAccess(ACCESS_READ_WRITE);
    for (int x = 0; x < output->getWidth(); x++){
        for (int y = 0; y < output->getHeight(); y++){
            for (int z = 0; z < output->getDepth(); z++){
                Vector3i location = Vector3i(x, y, z);
                float P = volAccess->getScalar(location, 0);
                float W = volAccess->getScalar(location, 1);
                if (W > 0.0 && P >= 0.0){ // W != 0.0 to avoid division error // This other logic to avoid -51.00 blanks
                    float finalP = P / W;
                    outAccess->setScalar(location, finalP, 0);
                }
                else{
                    outAccess->setScalar(location, 0.0, 0);
                }
            }
        }
    }
    std::cout << "Volume reconstructed!" << std::endl;
    outAccess.release();
    volAccess.release();
    
    //setStaticOutputData<Image>(0, output); //fails with this one? //3D
    // MAKE 2D Image cut
    outAccess = output->getImageAccess(ACCESS_READ);
    Image::pointer outputImg = getStaticOutputData<Image>(0);
    Vector3ui outputImgSize = output->getSize(); //Vector3i(output->getHeight(), output->getWidth(),1);
    outputImgSize(2) = 1;
    outputImg->create(outputImgSize, VoxelsValNWeight->getDataType(), 1);
    ImageAccess::pointer outImgAccess = outputImg->getImageAccess(ACCESS_READ_WRITE);
    for (int x = 0; x < output->getWidth(); x++){
        for (int y = 0; y < output->getHeight(); y++){
            Vector3i location = Vector3i(x, y, 0);
            //Vector2i outLoc = Vector2i(x, y);
            float p = outAccess->getScalar(location, 0);
            outImgAccess->setScalar(location, p, 0);
        }
    }
    outAccess.release();
    outImgAccess.release();
    setStaticOutputData<Image>(0, outputImg); //2D
    // MAKE 2D Image END
    std::cout << "Execute method finished succesfully!" << std::endl;
}
/*
void executeAlgorithmOnHost(Image::pointer input, Image::pointer output, float * mask, unsigned char maskSize) {
    // TODO: this method currently only processes the first component
    unsigned int nrOfComponents = input->getNrOfComponents();
    ImageAccess::pointer inputAccess = input->getImageAccess(ACCESS_READ);
    ImageAccess::pointer outputAccess = output->getImageAccess(ACCESS_READ_WRITE);

    T * inputData = (T*)inputAccess->get();
    T * outputData = (T*)outputAccess->get();

    const unsigned char halfSize = (maskSize - 1) / 2;
    unsigned int width = input->getWidth();
    unsigned int height = input->getHeight();
    if (input->getDimensions() == 3) {
        unsigned int depth = input->getDepth();
        for (unsigned int z = 0; z < depth; z++) {
            for (unsigned int y = 0; y < height; y++) {
                for (unsigned int x = 0; x < width; x++) {

                    if (x < halfSize || x >= width - halfSize ||
                        y < halfSize || y >= height - halfSize ||
                        z < halfSize || z >= depth - halfSize) {
                        // on border only copy values
                        outputData[x*nrOfComponents + y*nrOfComponents*width + z*nrOfComponents*width*height] = inputData[x*nrOfComponents + y*nrOfComponents*width + z*nrOfComponents*width*height];
                        continue;
                    }

                    double sum = 0.0;
                    for (int c = -halfSize; c <= halfSize; c++) {
                        for (int b = -halfSize; b <= halfSize; b++) {
                            for (int a = -halfSize; a <= halfSize; a++) {
                                sum += mask[a + halfSize + (b + halfSize)*maskSize + (c + halfSize)*maskSize*maskSize] *
                                    inputData[(x + a)*nrOfComponents + (y + b)*nrOfComponents*width + (z + c)*nrOfComponents*width*height];
                            }
                        }
                    }
                    outputData[x*nrOfComponents + y*nrOfComponents*width + z*nrOfComponents*width*height] = (T)sum;
                }
            }
        }
    }
    else {
        for (unsigned int y = halfSize; y < height - halfSize; y++) {
            for (unsigned int x = halfSize; x < width - halfSize; x++) {

                if (x < halfSize || x >= width - halfSize ||
                    y < halfSize || y >= height - halfSize) {
                    // on border only copy values
                    outputData[x*nrOfComponents + y*nrOfComponents*width] = inputData[x*nrOfComponents + y*nrOfComponents*width];
                    continue;
                }

                double sum = 0.0;
                for (int b = -halfSize; b <= halfSize; b++) {
                    for (int a = -halfSize; a <= halfSize; a++) {
                        sum += mask[a + halfSize + (b + halfSize)*maskSize] *
                            inputData[(x + a)*nrOfComponents + (y + b)*nrOfComponents*width];
                    }
                }
                outputData[x*nrOfComponents + y*nrOfComponents*width] = (T)sum;
            }
        }
    }
}*/

void PnnNoHf::initVolumeCube(Image::pointer rootFrame){
    /*
    bool dynData = input->isDynamicData();
    std::cout << "Is dynamic?" << dynData << std::endl;

    BoundingBox bb = input->getBoundingBox();
    std::cout << "Bounding box" << bb << std::endl;

    uint comps = input->getNrOfComponents();
    std::cout << "Components" << comps << std::endl;

    Vector3ui size = input->getSize();
    std::cout << "Size" << size << std::endl;
    //Vector3ui spacing = input->getSpacing();
    //std::cout << "Spacing" << spacing << std::endl;
    Streamer::pointer stream = input->getStreamer();
    std::cout << "Stream " << stream << std::endl;
    //uint nrOfFrames = stream->getNrOfFrames();
    */

    /*
    // Calculate image plane normal
    AffineTransformation::pointer imageTransformation = SceneGraph::getAffineTransformationFromData(rootFrame);
    Vector3f p0 = imageTransformation->multiply(Vector3f(0, 0, 0));
    Vector3f p1 = imageTransformation->multiply(Vector3f(1, 0, 0));
    Vector3f p2 = imageTransformation->multiply(Vector3f(0, 1, 0));
    Vector3f imagePlaneNormal = (p1 - p0).cross(p2 - p0);
    imagePlaneNormal.normalize();
    
    // Define zero point
    BoundingBox rootBB = rootFrame->getBoundingBox();
    uint width = rootFrame->getWidth();
    uint height = rootFrame->getHeight();
    uint components = rootFrame->getNrOfComponents();
    BoundingBox rootBBtrans = rootFrame->getTransformedBoundingBox();
    Eigen::Vector3f spacings = rootFrame->getSpacing();    

    Eigen::MatrixXf rootCorners = rootBB.getCorners();
    Eigen::MatrixXf rootBBtransCorners = rootBBtrans.getCorners();

    //Eigen::Vector3f cornerOne = rootCorners(0);
    std::vector<Eigen::Vector3f> points = {};
    for (int i = 0; i < 8; i++){
        Eigen::Vector3f corner(rootCorners(i, 0), rootCorners(i, 1), rootCorners(i, 2));
        Eigen::Vector3f cornerTrans(rootBBtransCorners(i, 0), rootBBtransCorners(i, 1), rootBBtransCorners(i, 2));
        points.push_back(cornerTrans);
    }
    */

    // BIG TODO
        // Find transform from rootFrame to normalized space
        // Use this transform to transform all frames
    // TO TRANSFORM: image->getSceneGraphNode()->setTransformation(transform), der transform er av typen AffineTransformation::pointer
    /*
    // Calculate image plane normal
    AffineTransformation::pointer imageTransformation = SceneGraph::getAffineTransformationFromData(rootFrame);
    Vector3f p0 = imageTransformation->multiply(Vector3f(0, 0, 0));
    Vector3f p1 = imageTransformation->multiply(Vector3f(1, 0, 0));
    Vector3f p2 = imageTransformation->multiply(Vector3f(0, 1, 0));
    Vector3f imagePlaneNormal = (p1 - p0).cross(p2 - p0);
    imagePlaneNormal.normalize();
    //AffineConst: AffineTransformation(const Eigen::Affine3f& transform);
        // Other: AffineTransformation::pointer multiply(AffineTransformation::pointer);
            //Vector3f multiply(Vector3f);
            //AffineTransformation& operator=(const Eigen::Affine3f& transform);
    //imageTransformation.inverse(TransformationTraits.Affine);
    /*Eigen::Transform<float, 3, Eigen::Affine> invTransformation = imageTransformation->inverse(); //Eigen::Transform<float,3,2,0>
    Eigen::Affine3f invTrans2 = invTransformation;
    AffineTransformation systemTransformation = invTrans2; //AffineTransformation::New();
    //systemTransformation::AffineTransformation(invTrans2);
    AffineTransformation::pointer systemTransformation2 = SharedPointer<AffineTransformation>(systemTransformation);*/
    AffineTransformation::pointer imageTransformation = SceneGraph::getAffineTransformationFromData(rootFrame);
    AffineTransformation::pointer inverseSystemTransform = imageTransformation->inverseTransform();
    Vector3f cornerBase = rootFrame->getBoundingBox().getCorners().row(0);
    Vector3f cornerTrans = rootFrame->getTransformedBoundingBox().getCorners().row(0);
    AffineTransformation::pointer rootTransform = imageTransformation->multiply(inverseSystemTransform);
    //New image transform = oldImageTransformation->multiply(inverseSystemTransform);
    rootFrame->getSceneGraphNode()->setTransformation(rootTransform); //and so on
    Vector3f cornerTransToBase = rootFrame->getTransformedBoundingBox().getCorners().row(0);

    // ## Calculate min-max in different directions
    Vector3f min, max;
    //Vector3f centroid;
    //BoundingBox box = rootFrame->getBoundingBox();
    BoundingBox box = rootFrame->getTransformedBoundingBox();
    MatrixXf corners = box.getCorners();
    Vector3f corner = box.getCorners().row(0);
    min[0] = corner[0]; //OR ()? TODO
    max[0] = corner[0];
    min[1] = corner[1];
    max[1] = corner[1];
    min[2] = corner[2];
    max[2] = corner[2];
    
    for (int i = 0; i < frameList.size(); i++){
        Image::pointer img = frameList[i];
        // Start transforming frame
        AffineTransformation::pointer oldImgTransform = SceneGraph::getAffineTransformationFromData(img);
        AffineTransformation::pointer newImgTransform = oldImgTransform->multiply(inverseSystemTransform);
        img->getSceneGraphNode()->setTransformation(newImgTransform);
        // Frame transformed get bounding box
        BoundingBox box = img->getTransformedBoundingBox();
        MatrixXf corners = box.getCorners();
        Vector3f baseCorner = corners.row(0);
        //TODO save each min/max for each frame in transformed directions!
        for (int j = 0; j < 8; j++) {
            for (uint k = 0; k < 3; k++) {
                float point = corners(j, k);
                if (point < min[k]) //corners(j, k) < min[k])
                    min[k] = point;
                if (point > max[k])
                    max[k] = point;
            }
        }
    }
    

    // Calculate directions relative in this one
    //  - x: horizontal; y: vertical in img; z: (0) in all of rootFrame

    // Init volume
    Vector3f size = max - min;
    volumeSize = Vector3i(ceil(size(0)), ceil(size(1)), ceil(size(2)));
    zeroPoints = min;
    DataType type = DataType::TYPE_FLOAT; //TYPE_INT8; //frame->getDataType();
    float initVal = 0.0;

    VoxelsValNWeight = Image::New();
    VoxelsValNWeight->create(volumeSize(0), volumeSize(1), volumeSize(2), type, 2);
    ImageAccess::pointer volAccess = VoxelsValNWeight->getImageAccess(accessType::ACCESS_READ_WRITE);
    for (int x = 0; x < volumeSize(0); x++){
        for (int y = 0; y < volumeSize(1); y++){
            for (int z = 0; z < volumeSize(2); z++){
                Vector3i location = Vector3i(x, y, z);
                float voxelValue0 = volAccess->getScalar(location, 0);
                volAccess->setScalar(location, initVal, 0); //Channel 1 - Value
                float voxelValue = volAccess->getScalar(location, 0);
                volAccess->setScalar(location, initVal, 1); //Channel 2 - Weight
                
                //imgAccess->setVector(Eigen::Vector3i(x, y, z), Eigen::Vector2i(0, 0)); // Eventuelt Vector2f etc
            }
        }
    }
    volAccess->release();

    /*volAccess = VoxelsValNWeight->getImageAccess(accessType::ACCESS_READ_WRITE);
    for (int x = 0; x < volumeSize[0]; x++){
        for (int y = 0; y < volumeSize[1]; y++){
            for (int z = 0; z < volumeSize[2]; z++){
                float voxelValue0 = volAccess->getScalar((x, y, z), 0);
                float voxelValue1 = volAccess->getScalar((x, y, z), 1);
            }
        }
    }
    volAccess->release();*/
    //calculate dv?
    int drrrrrrrr = 1;
}

void PnnNoHf::execute() {
    //Image::pointer input = getStaticInputData<Image>(0);

    Image::pointer frame = getStaticInputData<Image>(0);
    //Image::pointer output = getStaticOutputData<Image>(0);
    //output->createFromImage(frame);
    //setStaticOutputData<Image>(0, frame);
    //return;
    if (firstFrameNotSet){
        firstFrame = frame;
        //frameList.push_back(frame);
        firstFrameNotSet = false;

        /*DynamicData::pointer dynamicData = getInputData(0);
        dynamicData->registerConsumer(this);

        firstFrame = dynamicData->getNextFrame(this);
        frameList.push_back(firstFrame);
        output->createFromImage(firstFrame);

        while (!dynamicData->hasReachedEnd()){
        Image::pointer nextFrame = dynamicData->getNextFrame(this);
        frameList.push_back(nextFrame);
        }

        if (!volumeInitialized){
        std::cout << "INITIALIZING volume" << std::endl;
        //Init cube with all corners
        initVolumeCube(firstFrame);
        volumeInitialized = true;
        //Definer dv (oppløsning)
        dv = 1;
        }
        switch (firstFrame->getDataType()) {
        fastSwitchTypeMacro(executeAlgorithmOnHost<FAST_TYPE>(firstFrame, output));
        }*/

        //output = firstFrame;
        /*IMAGE constructors
        void create(VectorXui size, DataType type, uint nrOfComponents);
        void create(uint width, uint height, uint depth, DataType type, uint nrOfComponents);
        void create(VectorXui size, DataType type, uint nrOfComponents, ExecutionDevice::pointer device, const void * data);
        void create(uint width, uint height, uint depth, DataType type, uint nrOfComponents, ExecutionDevice::pointer device, const void * data);
        */
        //frame->getNrOfComponents;
        //frame->getDataType();
        //Image::pointer output = getStaticOutputData<Image>();
        //output->create(input->getSize(), TYPE_FLOAT, input->getNrOfComponents());


        output = getStaticOutputData<Image>(0);
        DataType type = DataType::TYPE_FLOAT; // INT8; //frame->getDataType();
        uint size = 32;
        //int initVal = 1;
        float initVal = 1.0;
        output->create(size, size, size, type, 1);// create(500, 500, 500, frame->getDataType(), 2);
        ImageAccess::pointer imgAccess = output->getImageAccess(accessType::ACCESS_READ_WRITE);
        ImageAccess::pointer inpAccess = frame->getImageAccess(accessType::ACCESS_READ);
        for (int x = 0; x < size; x++){
            for (int y = 0; y < size; y++){
                float thisVal = inpAccess->getScalar((x, y), 0);
                for (int z = 0; z < size; z++){
                    //imgAccess->setScalar((x, y, z), initVal, 0); //Channel 1 - Value
                    //imgAccess->setScalar((x, y, z), initVal, 1); //Channel 2 - Weight
                    imgAccess->setScalar((x, y, z), thisVal, 0);

                    //imgAccess->setVector(Eigen::Vector3i(x, y, z), Eigen::Vector2i(0, 0)); // Eventuelt Vector2f etc
                }
            }
        }
        imgAccess->release();
        inpAccess->release();
        std::cout << "MAX intensity" << output->calculateMaximumIntensity() << std::endl;

        //setStaticOutputData<Image>(0, output);

        //output->setSpacing(TODO);
    }
    // Lagre frame i PO, f.eks. i en std::vector

    //setStaticOutputData<Image>(0, output);
    setStaticOutputData<Image>(0, frame);

    frameList.push_back(frame);

    //Image::pointer output = getStaticOutputData<Image>();
    //output->create(256, 256, 256, frame->getDataType(), 2);
    // Sjekk om vi har nådd slutten
    DynamicData::pointer dynamicImage = getInputData(0);
    //dynamicImage->
    if (dynamicImage->hasReachedEnd()) {
        //Image::pointer output = getStaticOutputData<Image>(0);
        // Do reconstruction

        //output->setDimension(3);

        if (!volumeInitialized){
            std::cout << "INITIALIZING volume" << std::endl;
            //Init cube with all corners
            initVolumeCube(firstFrame);
            volumeInitialized = true;
            //Definer dv (oppløsning)
            dv = 1;
            // TODO initialize frame indexing!!! TODO
        }
        executeAlgorithmOnHost();// VoxelsValNWeight, output, frameList, dv, Rmax);
        /*switch (frame->getDataType()) {
        fastSwitchTypeMacro(executeAlgorithmOnHost<FAST_TYPE>(VoxelsValNWeight, output, frameList, dv, Rmax));
        }*/
    }
    /*else{// if (dynamicImage->getSize() == 0){
    std::cout << "DynImg size" << dynamicImage->getSize() << std::endl;
    }
    //getInputData(0);//getStaticInputData<Image>(0);
    /*if (input->getDimension() != 2){
    throw Exception("The algorithm only handles 2D image input");
    }*/
    //if (dynamicImage->)

    return;
    /*if (device->isHost()){
    switch (input->getDataType()) {
    fastSwitchTypeMacro(executeAlgorithmOnHost<FAST_TYPE>(input, output, mMask, maskSize));
    }
    }
    else{
    switch (input->getDataType()) {
    fastSwitchTypeMacro(executeAlgorithmOnHost<FAST_TYPE>(input, output, mMask, maskSize));
    }
    }

    /*
    char maskSize = mMaskSize;
    if (maskSize <= 0) // If mask size is not set calculate it instead
    maskSize = ceil(2 * mStdDev) * 2 + 1;

    if (maskSize > 19)
    maskSize = 19;

    // Initialize output image
    ExecutionDevice::pointer device = getMainDevice();
    if (mOutputTypeSet) {
    output->create(input->getSize(), mOutputType, input->getNrOfComponents());
    output->setSpacing(input->getSpacing());
    }
    else {
    output->createFromImage(input);
    }
    mOutputType = output->getDataType();
    SceneGraph::setParentNode(output, input);


    if (device->isHost()) {
    createMask(input, maskSize, false);
    switch (input->getDataType()) {
    fastSwitchTypeMacro(executeAlgorithmOnHost<FAST_TYPE>(input, output, mMask, maskSize));
    }
    }
    else {
    OpenCLDevice::pointer clDevice = device;

    recompileOpenCLCode(input);

    cl::NDRange globalSize;

    OpenCLImageAccess::pointer inputAccess = input->getOpenCLImageAccess(ACCESS_READ, device);
    if (input->getDimensions() == 2) {
    createMask(input, maskSize, false);
    mKernel.setArg(1, mCLMask);
    mKernel.setArg(3, maskSize);
    globalSize = cl::NDRange(input->getWidth(), input->getHeight());

    OpenCLImageAccess::pointer outputAccess = output->getOpenCLImageAccess(ACCESS_READ_WRITE, device);
    mKernel.setArg(0, *inputAccess->get2DImage());
    mKernel.setArg(2, *outputAccess->get2DImage());
    clDevice->getCommandQueue().enqueueNDRangeKernel(
    mKernel,
    cl::NullRange,
    globalSize,
    cl::NullRange
    );
    }
    else {
    // Create an auxilliary image
    Image::pointer output2 = Image::New();
    output2->createFromImage(output);

    globalSize = cl::NDRange(input->getWidth(), input->getHeight(), input->getDepth());

    if (clDevice->isWritingTo3DTexturesSupported()) {
    createMask(input, maskSize, true);
    mKernel.setArg(1, mCLMask);
    mKernel.setArg(3, maskSize);
    OpenCLImageAccess::pointer outputAccess = output->getOpenCLImageAccess(ACCESS_READ_WRITE, device);
    OpenCLImageAccess::pointer outputAccess2 = output2->getOpenCLImageAccess(ACCESS_READ_WRITE, device);

    cl::Image3D* image2;
    cl::Image3D* image;
    image = outputAccess->get3DImage();
    image2 = outputAccess->get3DImage();
    for (uchar direction = 0; direction < input->getDimensions(); ++direction) {
    if (direction == 0) {
    mKernel.setArg(0, *inputAccess->get3DImage());
    mKernel.setArg(2, *image);
    }
    else if (direction == 1) {
    mKernel.setArg(0, *image);
    mKernel.setArg(2, *image2);
    }
    else {
    mKernel.setArg(0, *image2);
    mKernel.setArg(2, *image);
    }
    mKernel.setArg(4, direction);
    clDevice->getCommandQueue().enqueueNDRangeKernel(
    mKernel,
    cl::NullRange,
    globalSize,
    cl::NullRange
    );
    }
    }
    else {
    createMask(input, maskSize, false);
    mKernel.setArg(1, mCLMask);
    mKernel.setArg(3, maskSize);
    OpenCLBufferAccess::pointer outputAccess = output->getOpenCLBufferAccess(ACCESS_READ_WRITE, device);
    mKernel.setArg(0, *inputAccess->get3DImage());
    mKernel.setArg(2, *outputAccess->get());
    clDevice->getCommandQueue().enqueueNDRangeKernel(
    mKernel,
    cl::NullRange,
    globalSize,
    cl::NullRange
    );
    }


    }
    }*/
}

void PnnNoHf::waitToFinish() {
    if (!getMainDevice()->isHost()) {
        OpenCLDevice::pointer device = getMainDevice();
        device->getCommandQueue().finish();
    }
}

/*
void GaussianSmoothingFilter::recompileOpenCLCode(Image::pointer input) {
    // Check if there is a need to recompile OpenCL code
    if(input->getDimensions() == mDimensionCLCodeCompiledFor &&
            input->getDataType() == mTypeCLCodeCompiledFor)
        return;

    OpenCLDevice::pointer device = getMainDevice();
    std::string buildOptions = "";
    if(!device->isWritingTo3DTexturesSupported()) {
        buildOptions = "-DTYPE=" + getCTypeAsString(mOutputType);
    }
    cl::Program program;
    if(input->getDimensions() == 2) {
        program = getOpenCLProgram(device, "2D", buildOptions);
    } else {
        program = getOpenCLProgram(device, "3D", buildOptions);
    }
    mKernel = cl::Kernel(program, "gaussianSmoothing");
    mDimensionCLCodeCompiledFor = input->getDimensions();
    mTypeCLCodeCompiledFor = input->getDataType();
}

template <class T>
void executeAlgorithmOnHost(Image::pointer input, Image::pointer output, float * mask, unsigned char maskSize) {
    // TODO: this method currently only processes the first component
    unsigned int nrOfComponents = input->getNrOfComponents();
    ImageAccess::pointer inputAccess = input->getImageAccess(ACCESS_READ);
    ImageAccess::pointer outputAccess = output->getImageAccess(ACCESS_READ_WRITE);

    T * inputData = (T*)inputAccess->get();
    T * outputData = (T*)outputAccess->get();

    const unsigned char halfSize = (maskSize-1)/2;
    unsigned int width = input->getWidth();
    unsigned int height = input->getHeight();
    if(input->getDimensions() == 3) {
        unsigned int depth = input->getDepth();
        for(unsigned int z = 0; z < depth; z++) {
        for(unsigned int y = 0; y < height; y++) {
        for(unsigned int x = 0; x < width; x++) {

            if(x < halfSize || x >= width-halfSize ||
            y < halfSize || y >= height-halfSize ||
            z < halfSize || z >= depth-halfSize) {
                // on border only copy values
                outputData[x*nrOfComponents+y*nrOfComponents*width+z*nrOfComponents*width*height] = inputData[x*nrOfComponents+y*nrOfComponents*width+z*nrOfComponents*width*height];
                continue;
            }

            double sum = 0.0;
            for(int c = -halfSize; c <= halfSize; c++) {
            for(int b = -halfSize; b <= halfSize; b++) {
            for(int a = -halfSize; a <= halfSize; a++) {
                sum += mask[a+halfSize+(b+halfSize)*maskSize+(c+halfSize)*maskSize*maskSize]*
                        inputData[(x+a)*nrOfComponents+(y+b)*nrOfComponents*width+(z+c)*nrOfComponents*width*height];
            }}}
            outputData[x*nrOfComponents+y*nrOfComponents*width+z*nrOfComponents*width*height] = (T)sum;
        }}}
    } else {
        for(unsigned int y = halfSize; y < height-halfSize; y++) {
        for(unsigned int x = halfSize; x < width-halfSize; x++) {

            if(x < halfSize || x >= width-halfSize ||
            y < halfSize || y >= height-halfSize) {
                // on border only copy values
                outputData[x*nrOfComponents+y*nrOfComponents*width] = inputData[x*nrOfComponents+y*nrOfComponents*width];
                continue;
            }

            double sum = 0.0;
            for(int b = -halfSize; b <= halfSize; b++) {
            for(int a = -halfSize; a <= halfSize; a++) {
                sum += mask[a+halfSize+(b+halfSize)*maskSize]*
                        inputData[(x+a)*nrOfComponents+(y+b)*nrOfComponents*width];
            }}
            outputData[x*nrOfComponents+y*nrOfComponents*width] = (T)sum;
        }}
    }
    
}*/


