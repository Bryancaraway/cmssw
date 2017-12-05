// -*- C++ -*-
//
// Package:    CalibTracker/SiStripESProducers
// Class:      SiStripBadModuleConfigurableFakeESSource
//
/**\class SiStripBadModuleConfigurableFakeESSource SiStripBadModuleConfigurableFakeESSource.h CalibTracker/SiStripESProducers/plugins/SiStripBadModuleConfigurableFakeESSource.cc

 Description: "fake" SiStripBadStrip ESProducer - configurable list of bad modules

 Implementation:
     Port of SiStripBadModuleGenerator and templated fake ESSource to an edm::ESProducer
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripBadModuleConfigurableFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripBadModuleConfigurableFakeESSource(const edm::ParameterSet&);
  ~SiStripBadModuleConfigurableFakeESSource() override;

  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity ) override;

  typedef std::shared_ptr<SiStripBadStrip> ReturnType;
  ReturnType produce(const SiStripBadModuleRcd&);

private:
  using Parameters = std::vector<edm::ParameterSet>;
  Parameters m_badComponentList;
  edm::FileInPath m_file;
  bool m_printDebug;

  std::vector<uint32_t> selectDetectors(const TrackerTopology* tTopo, const std::vector<uint32_t>& detIds) const;
};

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

SiStripBadModuleConfigurableFakeESSource::SiStripBadModuleConfigurableFakeESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<SiStripBadModuleRcd>();

  m_badComponentList = iConfig.getUntrackedParameter<Parameters>("BadComponentList");
  m_file = iConfig.getParameter<edm::FileInPath>("file");
  m_printDebug = iConfig.getUntrackedParameter<bool>("printDebug", false);
}

SiStripBadModuleConfigurableFakeESSource::~SiStripBadModuleConfigurableFakeESSource() {}

void SiStripBadModuleConfigurableFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity )
{
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

// ------------ method called to produce the data  ------------
SiStripBadModuleConfigurableFakeESSource::ReturnType
SiStripBadModuleConfigurableFakeESSource::produce(const SiStripBadModuleRcd& iRecord)
{
  using namespace edm::es;

  edm::ESHandle<TrackerTopology> tTopo;
  iRecord.getRecord<TrackerTopologyRcd>().get(tTopo);

  std::shared_ptr<SiStripQuality> quality{new SiStripQuality};

  SiStripDetInfoFileReader reader{m_file.fullPath()};

  std::vector<uint32_t> selDetIds{selectDetectors(tTopo.product(), reader.getAllDetIds())};
  edm::LogInfo("SiStripQualityConfigurableFakeESSource")<<"[produce] number of selected dets to be removed " << selDetIds.size() <<std::endl;

  std::stringstream ss;
  for ( const auto selId : selDetIds ) {
    SiStripQuality::InputVector theSiStripVector;

    unsigned short firstBadStrip{0};
    unsigned short NconsecutiveBadStrips = reader.getNumberOfApvsAndStripLength(selId).first * 128;
    unsigned int theBadStripRange{quality->encode(firstBadStrip,NconsecutiveBadStrips)};

    if (m_printDebug) {
      ss << "detid " << selId << " \t"
	 << " firstBadStrip " << firstBadStrip << "\t "
	 << " NconsecutiveBadStrips " << NconsecutiveBadStrips << "\t "
	 << " packed integer " << std::hex << theBadStripRange  << std::dec
	 << std::endl;
    }

    theSiStripVector.push_back(theBadStripRange);

    if ( ! quality->put(selId,SiStripBadStrip::Range{theSiStripVector.begin(),theSiStripVector.end()}) ) {
      edm::LogError("SiStripQualityConfigurableFakeESSource") << "[produce] detid already exists";
    }
  }
  if (m_printDebug) {
    edm::LogInfo("SiStripQualityConfigurableFakeESSource") << ss.str();
  }
  quality->cleanUp();
  //quality->fillBadComponents();

  if (m_printDebug){
    std::stringstream ss1;
    for ( const auto& badComp : quality->getBadComponentList() ) {
      ss1 << "bad module " << badComp.detid << " " << badComp.BadModule <<  "\n";
    }
    edm::LogInfo("SiStripQualityConfigurableFakeESSource") << ss1.str();
  }

  return quality;
}

namespace {
  bool _isSel( uint32_t requested, uint32_t i )
  { // internal helper: accept all i if requested is 0, otherwise require match
    return ( requested == 0 ) || ( requested == i );
  }

  SiStripDetId::SubDetector subDetFromString( const std::string& subDetStr )
  {
    SiStripDetId::SubDetector    subDet = SiStripDetId::UNKNOWN;
    if      ( subDetStr=="TIB" ) subDet = SiStripDetId::TIB;
    else if ( subDetStr=="TID" ) subDet = SiStripDetId::TID;
    else if ( subDetStr=="TOB" ) subDet = SiStripDetId::TOB;
    else if ( subDetStr=="TEC" ) subDet = SiStripDetId::TEC;
    return subDet;
  }
}

std::vector<uint32_t> SiStripBadModuleConfigurableFakeESSource::selectDetectors(const TrackerTopology* tTopo, const std::vector<uint32_t>& detIds) const
{
  std::vector<uint32_t> selList;
  std::stringstream ss;
  for ( const auto& badComp : m_badComponentList ) {
    const std::string subDetStr{badComp.getParameter<std::string>("SubDet")};
    if (m_printDebug) ss << "Bad SubDet " << subDetStr << " \t";
    const SiStripDetId::SubDetector subDet = subDetFromString(subDetStr);

    const std::vector<uint32_t> genericBadDetIds{badComp.getUntrackedParameter<std::vector<uint32_t>>("detidList", std::vector<uint32_t>())};
    const bool anySubDet{ ! genericBadDetIds.empty() };

    std::cout << "genericBadDetIds.size() = " << genericBadDetIds.size() << std::endl;

    using DetIdIt = std::vector<uint32_t>::const_iterator;
    const DetIdIt beginDetIt = std::lower_bound(detIds.begin(), detIds.end(), DetId(DetId::Tracker, anySubDet ? SiStripDetId::TIB   : subDet  ).rawId());
    const DetIdIt endDetIt   = std::lower_bound(detIds.begin(), detIds.end(), DetId(DetId::Tracker, anySubDet ? SiStripDetId::TEC+1 : subDet+1).rawId());

    if ( anySubDet ) {
      std::copy_if(beginDetIt, endDetIt, std::back_inserter(selList),
          [&genericBadDetIds] ( uint32_t detId ) {
            std::cout << "AnySubDet" << detId << std::endl;
            return std::find(genericBadDetIds.begin(), genericBadDetIds.end(), detId) != genericBadDetIds.end();
          });
    } else {
      switch (subDet) {
        case SiStripDetId::TIB:
          std::copy_if(beginDetIt, endDetIt, std::back_inserter(selList),
              [tTopo,&badComp] ( uint32_t detectorId ) {
                const DetId detId{detectorId};
                return ( ( detId.subdetId() == SiStripDetId::TIB )
                      && _isSel(badComp.getParameter<uint32_t>("layer")  , tTopo->tibLayer(detId))
                      && _isSel(badComp.getParameter<uint32_t>("bkw_frw"), tTopo->tibIsZPlusSide(detId) ? 2 : 1 )
                      && _isSel(badComp.getParameter<uint32_t>("int_ext"), tTopo->tibIsInternalString(detId) ? 1 : 2 )
                      && _isSel(badComp.getParameter<uint32_t>("ster")   , tTopo->tibIsStereo(detId) ? 1 : ( tTopo->tibIsRPhi(detId) ? 2 : -1 ))
                      && _isSel(badComp.getParameter<uint32_t>("string_"), tTopo->tibString(detId))
                      && _isSel(badComp.getParameter<uint32_t>("detid")  , detId.rawId())
                      );
              } );
          break;
        case SiStripDetId::TID:
          std::copy_if(beginDetIt, endDetIt, std::back_inserter(selList),
              [tTopo,&badComp] ( uint32_t detectorId ) {
                const DetId detId{detectorId};
                return ( ( detId.subdetId() ==  SiStripDetId::TID )
                      && _isSel(badComp.getParameter<uint32_t>("wheel"), tTopo->tidWheel(detId))
                      && _isSel(badComp.getParameter<uint32_t>("side") , tTopo->tidIsZPlusSide(detId) ? 2 : 1 )
                      && _isSel(badComp.getParameter<uint32_t>("ster") , tTopo->tidIsStereo(detId) ? 1 : ( tTopo->tidIsRPhi(detId) ? 2 : -1 ) )
                      && _isSel(badComp.getParameter<uint32_t>("ring") , tTopo->tidRing(detId))
                      && _isSel(badComp.getParameter<uint32_t>("detid"), detId.rawId())
                      );
              } );
          break;
        case SiStripDetId::TOB:
          std::copy_if(beginDetIt, endDetIt, std::back_inserter(selList),
              [tTopo,&badComp] ( uint32_t detectorId ) {
                const DetId detId{detectorId};
                return ( ( detId.subdetId() ==  SiStripDetId::TOB )
                      && _isSel(badComp.getParameter<uint32_t>("layer")  , tTopo->tobLayer(detId))
                      && _isSel(badComp.getParameter<uint32_t>("bkw_frw"), tTopo->tobIsZPlusSide(detId) ? 2 : 1)
                      && _isSel(badComp.getParameter<uint32_t>("ster")   , tTopo->tobIsStereo(detId) ? 1 : ( tTopo->tobIsRPhi(detId) ? 2 : -1 ))
                      && _isSel(badComp.getParameter<uint32_t>("rod")    , tTopo->tobRod(detId))
                      && _isSel(badComp.getParameter<uint32_t>("detid")  , detId.rawId())
                      );
              } );
          break;
        case SiStripDetId::TEC:
          std::copy_if(beginDetIt, endDetIt, std::back_inserter(selList),
              [tTopo,&badComp] ( uint32_t detectorId ) {
                const DetId detId{detectorId};
                return ( ( detId.subdetId() == SiStripDetId::TEC )
                      && _isSel(badComp.getParameter<uint32_t>("wheel")        , tTopo->tecWheel(detId))
                      && _isSel(badComp.getParameter<uint32_t>("side")         , tTopo->tecIsZPlusSide(detId) ? 2 : 1 )
                      && _isSel(badComp.getParameter<uint32_t>("ster")         , tTopo->tecIsStereo(detId) ? 1 : 2 )
                      && _isSel(badComp.getParameter<uint32_t>("petal_bkw_frw"), tTopo->tecIsFrontPetal(detId) ? 2 : 2)
                      && _isSel(badComp.getParameter<uint32_t>("petal")        , tTopo->tecPetalNumber(detId))
                      && _isSel(badComp.getParameter<uint32_t>("ring")         , tTopo->tecRing(detId))
                      && _isSel(badComp.getParameter<uint32_t>("detid")        , detId.rawId())
                      );
              } );
          break;
        default:
          break;
      }
    }
  }
  if (m_printDebug) {
    edm::LogInfo("SiStripBadModuleGenerator") << ss.str();
  }
  return selList;
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripBadModuleConfigurableFakeESSource);
