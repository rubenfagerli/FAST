/**
 * Examples/DataImport/streamImagesFromDisk.cpp
 *
 * If you edit this example, please also update the wiki and source code file in the repository.
 */
#include "FAST/Streamers/ImageFileStreamer.hpp"
#include "FAST/Visualization/ImageRenderer/ImageRenderer.hpp"
#include "FAST/Visualization/VolumeRenderer/VolumeRenderer.hpp"
#include "FAST/Visualization/SimpleWindow.hpp"
#include "FAST/TestDataPath.hpp"
#include "FAST/Algorithms/UsReconstruction/PnnNoHf.hpp"

using namespace fast;

int main() {
    // Import images from files using the ImageFileStreamer
    ImageFileStreamer::pointer streamer = ImageFileStreamer::New();
    // The hashtag here will be replaced with an integer, starting with 0 as default
    //std::string folder = 'US-2Dt';
    //std::nameformat = 'US-2Dt_#.mhd';

    std::string folder = "/rekonstruksjons_data/US_01_20130529T084519/";
    std::string nameformat = "US_01_20130529T084519_ScanConverted_#.mhd";
    
    streamer->setStreamingMode(STREAMING_MODE_PROCESS_ALL_FRAMES);
    //streamer->setStreamingMode(STREAMING_MODE_STORE_ALL_FRAMES);
    streamer->setFilenameFormat(std::string(FAST_TEST_DATA_DIR)+folder+nameformat);
    //streamer->setMaximumNumberOfFrames(746); //746 total
    streamer->setStartNumber(735);//200);
    //streamer->setStepSize(2);
    //streamer->enableLooping();
    std::cout << "Nr of frames" << streamer->getNrOfFrames() << std::endl;

    // Reconstruction PNN
    PnnNoHf::pointer pnn = PnnNoHf::New();
    pnn->setInputConnection(streamer->getOutputPort());

    // Renderer volume
    VolumeRenderer::pointer volumeRenderer = VolumeRenderer::New();
    volumeRenderer->addInputConnection(pnn->getOutputPort());
    OpacityTransferFunction::pointer otf = OpacityTransferFunction::New();
    otf->addAlphaPoint(0.0, 0.0);
    otf->addAlphaPoint(1.0, 0.5);
    ColorTransferFunction::pointer ctf = ColorTransferFunction::New();
    ctf->addRGBPoint(0.0, 0, 1, 0);
    ctf->addRGBPoint(1.0, 1, 0, 0);
    
    volumeRenderer->setColorTransferFunction(0, ctf);
    volumeRenderer->setOpacityTransferFunction(0, otf);

    //Alt for now, display image
    //Image renderer
    ImageRenderer::pointer imageRenderer = ImageRenderer::New();
    imageRenderer->addInputConnection(pnn->getOutputPort());

    SimpleWindow::pointer window = SimpleWindow::New();
    window->addRenderer(imageRenderer); //volumeRenderer);//renderer);
    //window->setMaximumFramerate(10); //unngå at buffer går tomt?
    //window->set2DMode();
    window->setTimeout(60*1000); // automatically close window after 5 seconds
    window->start();
}