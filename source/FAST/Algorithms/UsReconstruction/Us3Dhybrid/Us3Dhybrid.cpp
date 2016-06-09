#include "FAST/Algorithms/UsReconstruction/Us3Dhybrid/Us3Dhybrid.hpp"
#include "FAST/Exception.hpp"
#include "FAST/DeviceManager.hpp"
#include "FAST/Data/Image.hpp"
using namespace fast;

void Us3Dhybrid::setOutputType(DataType type){
    mOutputType = type;
    mOutputTypeSet = true;
    mIsModified = true;
}

Us3Dhybrid::Us3Dhybrid(){
    createInputPort<Image>(0);
    createOutputPort<Image>(0, OUTPUT_DEPENDS_ON_INPUT, 0); //OUTPUT_DEPENDS_ON_INPUT necessary? TODO
    //create openCL prog here
    //--//createOpenCLProgram(std::string(FAST_SOURCE_DIR) + "Algorithms/GaussianSmoothingFilter/GaussianSmoothingFilter2D.cl", "2D");
    //--// store different compiled for settings (dimension/variables...)
    mIsModified = true; // needed?
    mOutputTypeSet = false;

    //volume;
    dv = 1.0;
    Rmax = 3.0; //2?
    volumeCalculated = false;
    volumeInitialized = false;
    firstFrameSet = false;
    reachedEndOfStream = false;
    frameList = {};
    iterartorCounter = 0;
}

Us3Dhybrid::~Us3Dhybrid(){
    //delete something
}

void Us3Dhybrid::waitToFinish() {
    if (!getMainDevice()->isHost()) {
        OpenCLDevice::pointer device = getMainDevice();
        device->getCommandQueue().finish();
    }
}

void Us3Dhybrid::accumulateValuesInVolume(Vector3i volumePoint, float p, float w){
    //volAccess available from Us3Dhybrid as ImageAccess::pointer
    float oldP = volAccess->getScalar(volumePoint, 0);
    float oldW = volAccess->getScalar(volumePoint, 1);
    float newP = oldP + p*w;
    float newW = oldW + w;
    volAccess->setScalar(volumePoint, newP, 0);
    volAccess->setScalar(volumePoint, newW, 1);
}

// CPU algoritme
//template <class T>
void Us3Dhybrid::executeAlgorithmOnHost(){
    //Get access to volume on which we accumulate the values in
    // (volAccess is defined globally in Us3Dhybrid as an ImageAccess::pointer)
    volAccess = AccumulationVolume->getImageAccess(accessType::ACCESS_READ_WRITE);
    //Vector3ui volumeSize = AccumulationVolume->getSize(); //TODO implement proper //Todo make global?
    // For each FRAME
    for (int frameNr = 0; frameNr < frameList.size(); frameNr++){
        // Get FRAME
        Image::pointer frame = frameList[frameNr];
        // Calc imagePlaneNormal and dominating direction of it
        Vector3f imagePlaneNormal = getImagePlaneNormal(frame);
        int domDir = getDominatingVectorDirection(imagePlaneNormal);
        float domVal = fabs(imagePlaneNormal(domDir));

        // Get current, last and next plane
        // TODO define Vector4f? ax+by+cz+d=0? eller normal vector + point?
        // Defining plane by normalVector and the world coordinate of the (0,0) pixel point
        // # thisFrameRootPoint, 
        //   # thisFrameNormal = imagePlaneNormal
        //   # thisFrameSize
        // # lastFrameRootPoint, lastFrameNormal
        // # nextFrameRootPoint, nextFrameNormal
        // TODO fix storage and fetching of these
        Vector3f thisFrameRootPoint = Vector3f(0, 0, 0);
        Vector3f lastFrameRootPoint = Vector3f(0, 0, 0);
        Vector3f lastFrameNormal = Vector3f(0, 0, 0);
        Vector3f nextFrameRootPoint = Vector3f(0, 0, 0);
        Vector3f nextFrameNormal = Vector3f(0, 0, 0);

        // Get frame access
        ImageAccess::pointer frameAccess = frame->getImageAccess(accessType::ACCESS_READ);

        // Find size of non-dominating directions in volume space (a-dir & b-dir)
        Vector2i aDirRange = getFrameRangeInVolume(frame, domDir, 0); //a: 0
        Vector2i bDirRange = getFrameRangeInVolume(frame, domDir, 1); //b: 1

        //For each a in a-dir
        for (int a = aDirRange(0); a <= aDirRange(1); a++){
            //For each b in b-dir
            for (int b = bDirRange(0); b <= bDirRange(1); b++){
                //Find basePoint in the plane based on the a and b values
                Vector3f basePoint = getBasePointInPlane(thisFrameRootPoint, imagePlaneNormal, a, b, domDir);
                //Find distance to last and next frame
                float d1 = getDistanceAlongNormal(basePoint, imagePlaneNormal, lastFrameRootPoint, lastFrameNormal);
                float d2 = getDistanceAlongNormal(basePoint, imagePlaneNormal, nextFrameRootPoint, nextFrameNormal);
                //Calculate half width df and dfDom
                float df = calculateHalfWidth(d1, d2, dv, Rmax);
                float dfDom = df / domVal;

                //Indeks for c-dir range in domDir
                /*Vector2i cDirRange = getDomDirRange(basePoint, domDir, dfDom, volumeSize);
                //For hver c i c-dir
                for (int c = cDirRange(0); c <= cDirRange(1); c++){
                Vector3i volumePoint = getVolumePointLocation(a, b, c, domDir);
                //TODO implement
                /*Vector3f intersectionPointWorld = getIntersectionOfPlane(volumePoint, thisFrameRootPoint, imagePlaneNormal);
                Vector3f intersectionPointLocal = getLocalIntersectionOfPlane(); //TODO from what?
                if (intersectionWithinFrame(frame, intersectionPointLocal)){ //Or check through something else
                // Calculate pixelvalue p and weight w
                float p = getPixelValue(frameAccess, intersectionPointLocal);
                float distance = getPointDistanceAlongNormal(volumePoint, intersectionPointWorld, imagePlaneNormal);
                float w = 1 - distance / df; //Or gaussian for trail
                accumulateValuesInVolume(volumePoint, p, w);
                }*
                }*/
            }
        }
    }

    // Finally, calculate reconstructed volume
    output = getStaticOutputData<Image>(0);
    output->create(AccumulationVolume->getSize(), AccumulationVolume->getDataType(), 1); //1-channeled output volume
    ImageAccess::pointer outAccess = output->getImageAccess(ACCESS_READ_WRITE);
    for (int x = 0; x < output->getWidth(); x++){
        for (int y = 0; y < output->getHeight(); y++){
            for (int z = 0; z < output->getDepth(); z++){
                Vector3i location = Vector3i(x, y, z);
                float p = volAccess->getScalar(location, 0);
                float w = volAccess->getScalar(location, 1);
                if (w > 0.0 && p >= 0.0){ // w != 0.0 to avoid division error // This other logic to avoid uninitialized voxels
                    float finalP = p / w;
                    outAccess->setScalar(location, finalP, 0);
                }
                else{
                    outAccess->setScalar(location, 0.0, 0);
                }
            }
        }
    }
    outAccess.release();

    //Can possibly make 2D slices here or alternatively to the one above
    volAccess.release();
}

void Us3Dhybrid::initVolume(Image::pointer rootFrame){
    //Using object-defined variables firstFrame, frameList and possible preset variables
    //Find initial transform so that firstFrame corner(0,0) is voxel(0,0,0) in volume 
    //and all pixels in this frame would be in the z=0 plane
    AffineTransformation::pointer imageTransformation = SceneGraph::getAffineTransformationFromData(rootFrame);
    AffineTransformation::pointer inverseSystemTransform = imageTransformation->inverseTransform();
    AffineTransformation::pointer rootTransform = imageTransformation->multiply(inverseSystemTransform);
    //Transform all frames according to initial transform
    // & Find min/max in each coordinate direction x/y/z
    // & Store min/max/base/normal for each frame
    Vector3f minCoords, maxCoords;
    BoundingBox box = rootFrame->getTransformedBoundingBox();
    Vector3f corner = box.getCorners().row(0);
    minCoords(0) = corner(0);
    maxCoords(0) = corner(0);
    minCoords(1) = corner(1);
    maxCoords(1) = corner(1);
    minCoords(2) = corner(2);
    maxCoords(2) = corner(2);
    //Define lists to store results for each frame
    frameMinList = {}; // std::vector global Vector3f
    frameMaxList = {}; // std::vector global Vector3f
    frameBaseCornerList = {}; // std::vector global Vector3f
    framePlaneNormalList = {}; // std::vector global Vector3f
    for (int i = 0; i < frameList.size(); i++){
        Image::pointer frame = frameList[i];
        // Start transforming frame
        AffineTransformation::pointer oldImgTransform = SceneGraph::getAffineTransformationFromData(frame);
        AffineTransformation::pointer newImgTransform = oldImgTransform->multiply(inverseSystemTransform);
        frame->getSceneGraphNode()->setTransformation(newImgTransform);
        // Check corners min/max of frame
        BoundingBox box = frame->getTransformedBoundingBox();
        MatrixXf corners = box.getCorners();
        Vector3f baseCorner = corners.row(0);
        Vector3f minCoordsFrame = baseCorner;
        Vector3f maxCoordsFrame = baseCorner;
        for (int j = 0; j < 8; j++){
            for (int k = 0; k < 3; k++){
                float pointValue = corners(j, k);
                if (pointValue < minCoordsFrame(k))
                    minCoordsFrame(k) = pointValue;
                if (pointValue > maxCoordsFrame(k))
                    maxCoordsFrame(k) = pointValue;
            }
        }
        for (int k = 0; k < 3; k++){
            if (minCoordsFrame(k) < minCoords(k))
                minCoords(k) = minCoordsFrame(k);
            if (maxCoordsFrame(k) < maxCoords(k))
                maxCoords(k) = maxCoordsFrame(k);
        }
        // Store frame values for later
        frameMinList.push_back(minCoordsFrame);
        frameMaxList.push_back(maxCoordsFrame);
        frameBaseCornerList.push_back(baseCorner);
        framePlaneNormalList.push_back(getImagePlaneNormal(frame));
    }
    //Transform all frames so that minimum corner is (0,0,0) //Just translate right?
    /* // BIG TODO FIX THIS PART TODO TODO
    AffineTransformation::pointer transformToMinimum = getTransformationFromVector3f(minCoords); //TODO extract these to methods
    for (int i = 0; i < frameList.size(); i++){
    Image::pointer frame = frameList[i];
    // Start transforming frame
    AffineTransformation::pointer oldImgTransform = SceneGraph::getAffineTransformationFromData(frame);
    AffineTransformation::pointer newImgTransform = oldImgTransform->multiply(inverseSystemTransform); //TODO Change to translation !!!!!!!!!!!!!!!
    frame->getSceneGraphNode()->setTransformation(newImgTransform);
    }*/
    //Init volume of size max-min in each direction x/y/z
    Vector3f size = maxCoords - minCoords;
    volumeSize = Vector3i(ceil(size(0)), ceil(size(1)), ceil(size(2)));
    DataType type = DataType::TYPE_FLOAT; //Endre til INT p� sikt?
    float initVal = 0.0;
    int components = 2; // pixelvalues & weights
    AccumulationVolume = Image::New();
    AccumulationVolume->create(volumeSize(0), volumeSize(1), volumeSize(2), type, components);
    //Init volume to zero values and two components
    volAccess = AccumulationVolume->getImageAccess(accessType::ACCESS_READ_WRITE); //global volAccess ImageAccess::pointer
    for (int x = 0; x < volumeSize(0); x++){
        for (int y = 0; y < volumeSize(1); y++){
            for (int z = 0; z < volumeSize(2); z++){
                Vector3i location = Vector3i(x, y, z);
                volAccess->setScalar(location, initVal, 0); //Channel 1 - Value
                volAccess->setScalar(location, initVal, 1); //Channel 2 - Weight
            }
        }
    }
    volAccess->release();

    //Init dv (based on input frames/userdefined settings?)
    //TODO
}

void Us3Dhybrid::execute(){
    if (!reachedEndOfStream){
        std::cout << "Iteration #:" << iterartorCounter++ << std::endl;
        Image::pointer frame = getStaticInputData<Image>(0);
        frameList.push_back(frame);
        if (!firstFrameSet){
            firstFrame = frame;
            firstFrameSet = true;
        }
        // Sjekk om vi har n�dd slutten
        DynamicData::pointer dynamicImage = getInputData(0);
        if (dynamicImage->hasReachedEnd()) {
            reachedEndOfStream = true;
        }
        setStaticOutputData<Image>(0, frame);
    }
    // When we have reached the end of stream we do just from here on
    if (reachedEndOfStream) {
        std::cout << "END Iteration #:" << iterartorCounter++ << std::endl;
        if (!volumeCalculated){
            if (!volumeInitialized){
                std::cout << "Nr of frames in frameList:" << frameList.size() << std::endl;
                std::cout << "INITIALIZING volume" << std::endl;
                //Init cube with all corners
                initVolumeCube(firstFrame);
                volumeInitialized = true;
                //Definer dv (oppl�sning)
                dv = 1; //ev egen function to define DV
                //outputImg = firstFrame;
            }
            //if use GPU else :
            executeAlgorithmOnHost();
        }
        setStaticOutputData<Image>(0, output);
    }
}

// ##### ##### Other functions ##### ##### //
Vector3f getImagePlaneNormal(Image::pointer frame){
    AffineTransformation::pointer imageTransformation = SceneGraph::getAffineTransformationFromData(frame);
    Vector3f p0 = imageTransformation->multiply(Vector3f(0, 0, 0));
    Vector3f p1 = imageTransformation->multiply(Vector3f(1, 0, 0));
    Vector3f p2 = imageTransformation->multiply(Vector3f(0, 1, 0));
    Vector3f imagePlaneNormal = (p1 - p0).cross(p2 - p0);
    imagePlaneNormal.normalize();
    return imagePlaneNormal;
}

Vector3i getRoundedIntVector3f(Vector3f v){
    return Vector3i(round(v(0)), round(v(1)), round(v(2)));
}

int getDominatingVectorDirection(Vector3f v){
    float domVal = fabs(v(0));
    int domDir = 0;
    if (fabs(v(1)) > domVal){
        domVal = fabs(v(1));
        domDir = 1;
    }
    if (fabs(v(2)) > domVal){
        domVal = fabs(v(2));
        domDir = 2;
    }
    return domDir;
}

Vector2i getFrameRangeInVolume(Image::pointer frame, int domDir, int dir){
    //domDir of x,y,z and dir of a,b
    Vector2i output;
    //If square just use corners //TODO implement alternative?
    Vector3f minCoords;
    Vector3f maxCoords;
    //TODO implement or fetch minmax
    if ((domDir == 0 && dir == 0) || (domDir == 2 && dir == 1)){
        //If domDir:x want a-dir or if domDir:z want b-dir
        //We are returning range from the Y-axis
        //TODO
    }
    else if ((domDir == 1 && dir == 0) || (domDir == 2 && dir == 0)){
        //If domDir:y or domDir:z and want a-dir
        //We are returning range from the X-axis
    }
    else if ((domDir == 0 && dir == 1) || (domDir == 1 && dir == 1)){
        //If domDir:x or domDir:y and want b-dir
        //We are returning range from the Z-axis
    }
    return output;
}

Vector2i getDomDirRange(Vector3f basePoint, int domDir, float dfDom, Vector3i volumeSize){
    float rootC = basePoint(domDir);
    int domDirSize = volumeSize(domDir);
    int startC = std::max(0.0f, ceil(rootC - dfDom));
    int endC = std::min(ceil(rootC + dfDom), float(domDirSize));
    return Vector2i(startC, endC);
}



//Seems good?
Vector3f getBasePointInPlane(Vector3f rootPoint, Vector3f normal, int a, int b, int domDir){
    float x, y, z;
    if (domDir == 0){ //domDir: x
        y = a;
        z = b;
        //x
        if (normal(0) != 0.0){
            x = -(normal(1)*y + normal(2)*z) / normal(0); //d utelatt?
            //TODO finn ut om vi kan droppe d i fra planligningen
        }
    }
    else if (domDir == 1){ //domDir: y
        x = a;
        z = b;
        //y
        if (normal(1) != 0.0){
            y = -(normal(0)*x + normal(2)*z) / normal(1); //d utelatt?
        }
    }
    else if (domDir == 2){ //domDir: z
        x = a;
        y = b;
        //z
        if (normal(2) != 0.0){
            z = -(normal(0)*x + normal(1)*y) / normal(2); //d utelatt?
        }
    }
    return Vector3f(x, y, z);
}

float getPointDistanceAlongNormal(Vector3i A, Vector3f B, Vector3f normal){
    // |(B-A).dot(normal)|
    // TODO check maths is performed correctly
    //float distance = Vector3f((B - A).dot(normal)).norm();
    Vector3f Af = Vector3f(A(0), A(1), A(2));
    Vector3f diff = (B - Af);
    //Vector3f diff = Vector3f()
    float prod = diff.dot(normal);
    float distance = fabs(prod); //fabs((B - A).dot(normal));
    return distance;
}

/*
Get distance from point worldPoint to plane neighFrame along the imagePlaneNormal //TODO update doc
* point: Point in world space
* normal: Normal vector from worldPoint
* planePoint & planeNormal: Point and normal used to define a neighboring frame plane
# Return distance from point to plane along normal
*/
float getDistanceAlongNormal(Vector3f point, Vector3f normal, Vector3f planePoint, Vector3f planeNormal){
    //P0 = planePoint
    //L0 = point in world
    //N = planeNormal
    //L = normal from point/origin
    float divisor = normal.dot(planeNormal);
    float dividend = (point - planePoint).dot(planeNormal);
    if (divisor == 0.0){
        if (dividend == 0.0){
            //Parallell with the plane, IN the plane
            return 0.0;
        }
        else {
            //Parallell with the plane, but NOT in it
            return 1000.0; //OR Infinite?
        }
    }
    else {

        float distance = dividend / divisor;
        return distance;
    }
}

//GOOD
Vector3i getVolumePointLocation(int a, int b, int c, int domDir){
    int x, y, z;
    if (domDir == 0){ //domDir: x
        x = c;
        y = a;
        z = b;
    }
    else if (domDir == 1){ //domDir: y
        x = a;
        y = c;
        z = b;
    }
    else if (domDir == 2){ //domDir: z
        x = a;
        y = b;
        z = c;
    }
    return Vector3i(x, y, z);
}

float calculateHalfWidth(float d1, float d2, float dv, float Rmax){
    float furthestNeighbour = std::max(d1, d2);
    float maxTotal = std::max(furthestNeighbour, dv);
    float results = std::min(maxTotal, Rmax);
    return results;
}


