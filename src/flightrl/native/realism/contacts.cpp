// Jolt integration follows the MIT-licensed upstream HelloWorld interfaces.
#include "world.h"
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <algorithm>
#include <cmath>
#include <cstring>

static thread_local std::string error;
static Vec3 vec(const float *p) { return Vec3(p[0], p[1], p[2]); }
static Quat quat(const float *q) { return Quat(q[0], q[1], q[2], q[3]); }
extern "C" {
const char *fr_error() { return error.c_str(); }
World *fr_create() {
    static std::once_flag initialized;
    std::call_once(initialized, [] { RegisterDefaultAllocator(); Factory::sInstance = new Factory(); RegisterTypes(); });
    return new World();
}
void fr_destroy(World *world) { delete world; }
int fr_mesh(World *w, const float *vertices, int nv, const unsigned *indices, int nt) {
    VertexList points; points.reserve(nv);
    for (int i = 0; i < nv; i++) points.emplace_back(vertices[3*i], vertices[3*i+1], vertices[3*i+2]);
    IndexedTriangleList triangles; triangles.reserve(nt);
    for (int i = 0; i < nt; i++) triangles.emplace_back(indices[3*i], indices[3*i+1], indices[3*i+2]);
    MeshShapeSettings settings(points, triangles);
    auto shape = settings.Create();
    if (shape.HasError()) { error = shape.GetError().c_str(); return -1; }
    BodyCreationSettings body(shape.Get(), RVec3::sZero(), Quat::sIdentity(), EMotionType::Static, 0);
    body.mFriction = .65f; body.mRestitution = .08f;
    return w->add(body);
}
int fr_box(World *w, const float *position, const float *rotation, const float *half, float mass) {
    auto shape = BoxShapeSettings(vec(half), std::min(.002f, .1f * std::min({half[0],half[1],half[2]}))).Create();
    if (shape.HasError()) { error = shape.GetError().c_str(); return -1; }
    BodyCreationSettings body(shape.Get(), RVec3(vec(position)), quat(rotation), EMotionType::Dynamic, 1);
    body.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
    body.mMassPropertiesOverride.mMass = mass;
    body.mMotionQuality = EMotionQuality::LinearCast;
    body.mFriction = .65f; body.mRestitution = .08f;
    body.mLinearDamping = 0; body.mAngularDamping = 0;
    return w->add(body);
}
int fr_step(World *w, float dt) {
    w->contacts.impacts.clear();
    return int(w->physics.Update(dt, 1, &w->allocator, &w->jobs));
}
void fr_force(World *w, int id, const float *force) { w->physics.GetBodyInterface().AddForce(w->ids.at(id), vec(force)); }
void fr_velocity(World *w, int id, const float *v, const float *omega) {
    w->physics.GetBodyInterface().SetLinearAndAngularVelocity(w->ids.at(id), vec(v), vec(omega));
}
void fr_angular(World *w, int id, const float *omega) { w->physics.GetBodyInterface().SetAngularVelocity(w->ids.at(id), vec(omega)); }
void fr_transform(World *w, int id, const float *p, const float *q) {
    w->physics.GetBodyInterface().SetPositionAndRotation(w->ids.at(id), RVec3(vec(p)), quat(q), EActivation::Activate);
}
void fr_state(World *w, int id, float *out) {
    auto &api = w->physics.GetBodyInterface(); auto body = w->ids.at(id);
    auto p = api.GetPosition(body); auto q = api.GetRotation(body);
    auto v = api.GetLinearVelocity(body); auto r = api.GetAngularVelocity(body);
    float data[] = {float(p.GetX()),float(p.GetY()),float(p.GetZ()),q.GetX(),q.GetY(),q.GetZ(),q.GetW(),
        v.GetX(),v.GetY(),v.GetZ(),r.GetX(),r.GetY(),r.GetZ()};
    std::memcpy(out, data, sizeof(data));
}
int fr_contacts(World *w, float *out, int capacity) {
    int n = std::min(capacity, int(w->contacts.impacts.size()));
    if (n) std::memcpy(out, w->contacts.impacts.data(), n * sizeof(Impact));
    return n;
}
void fr_rays(World *w, const float *starts, const float *directions, int count, float length, float *out) {
    for (int i = 0; i < count; i++) {
        RRayCast ray(RVec3(vec(starts+3*i)), vec(directions+3*i)*length);
        RayCastResult result; Vec3 normal = Vec3::sZero();
        bool hit = w->physics.GetNarrowPhaseQuery().CastRay(ray, result);
        if (hit) {
            BodyLockRead lock(w->physics.GetBodyLockInterface(), result.mBodyID);
            if (lock.Succeeded()) normal = lock.GetBody().GetWorldSpaceSurfaceNormal(result.mSubShapeID2, ray.GetPointOnRay(result.mFraction));
        }
        out[5*i] = hit ? result.mFraction : 1;
        out[5*i+4] = -1;
        if (hit) for (size_t j = 0; j < w->ids.size(); ++j)
            if (w->ids[j] == result.mBodyID) { out[5*i+4] = float(j); break; }
        out[5*i+1] = normal.GetX(); out[5*i+2] = normal.GetY(); out[5*i+3] = normal.GetZ();
    }
}
}
