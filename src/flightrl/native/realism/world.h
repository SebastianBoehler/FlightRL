#pragma once
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/Body.h>
#include <mutex>
#include <vector>
#include <string>
using namespace JPH;

struct Layers final : BroadPhaseLayerInterface {
    uint GetNumBroadPhaseLayers() const override { return 2; }
    BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer layer) const override { return BroadPhaseLayer(layer); }
};
struct Pairs final : ObjectLayerPairFilter {
    bool ShouldCollide(ObjectLayer a, ObjectLayer b) const override { return a || b; }
};
struct BroadPairs final : ObjectVsBroadPhaseLayerFilter {
    bool ShouldCollide(ObjectLayer a, BroadPhaseLayer b) const override { return a || b.GetValue(); }
};
struct Impact { float x, y, z, speed; };
struct Contacts final : ContactListener {
    std::vector<Impact> impacts;
    void OnContactAdded(const Body &a, const Body &b, const ContactManifold &m, ContactSettings &) override {
        auto p = m.GetWorldSpaceContactPointOn1(0);
        float speed = (a.GetPointVelocity(p) - b.GetPointVelocity(p)).Length();
        impacts.push_back({float(p.GetX()), float(p.GetY()), float(p.GetZ()), speed});
    }
};
struct World {
    Layers layers; Pairs pairs; BroadPairs broad;
    TempAllocatorImpl allocator{16 * 1024 * 1024};
    JobSystemSingleThreaded jobs{cMaxPhysicsJobs};
    Contacts contacts;
    PhysicsSystem physics;
    std::vector<BodyID> ids;
    World() {
        physics.Init(128, 0, 4096, 2048, layers, broad, pairs);
        physics.SetGravity(Vec3(0, 0, -9.81f));
        auto settings = physics.GetPhysicsSettings();
        settings.mPenetrationSlop = .001f;
        settings.mSpeculativeContactDistance = .005f;
        settings.mNumVelocitySteps = 12;
        settings.mNumPositionSteps = 4;
        physics.SetPhysicsSettings(settings);
        physics.SetContactListener(&contacts);
    }
    int add(const BodyCreationSettings &settings) {
        BodyID id = physics.GetBodyInterface().CreateAndAddBody(settings, EActivation::Activate);
        if (id.IsInvalid()) return -1;
        ids.push_back(id); return int(ids.size() - 1);
    }
};
