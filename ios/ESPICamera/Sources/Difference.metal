// The on-phone difference view: |image(n) - image(n-1)| in the source's own
// levels, written as display grey (the source's full scale maps to white).
// Keep DiffParams in step with DifferenceEngine.swift.

#include <metal_stdlib>
using namespace metal;

struct DiffParams {
    uint offsetX;      // source pixel read for output (0, 0): the red Bayer site
    uint offsetY;
    uint step;         // source pixels per output pixel (2 for Bayer, 1 for luma)
    uint shift;        // right shift to the sample's own levels (6 for x420)
    uint maxLevel;     // 2^bits - 1, shown as white
    uint hasPrevious;  // 0: no earlier image, so the output is black
};

kernel void raw_difference(texture2d<ushort, access::read> source [[texture(0)]],
                           texture2d<ushort, access::read> previous [[texture(1)]],
                           texture2d<ushort, access::write> current [[texture(2)]],
                           texture2d<float, access::write> out [[texture(3)]],
                           constant DiffParams &p [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    uint2 site = gid * p.step + uint2(p.offsetX, p.offsetY);
    uint now = uint(source.read(site).r) >> p.shift;
    current.write(ushort4(ushort(now)), gid);
    uint d = 0;
    if (p.hasPrevious != 0) {
        d = uint(abs(int(now) - int(previous.read(gid).r)));
    }
    uint grey = min((d * 255u + (p.maxLevel >> 1)) / p.maxLevel, 255u);
    out.write(float4(float(grey) / 255.0), gid);
}

// Drawing the difference on screen: a quad of half-extent `extent` (clip
// space), with the landscape image turned 90 degrees clockwise to portrait,
// as the camera preview is. Linear filtering is for display only.

struct DisplayVertex {
    float4 position [[position]];
    float2 uv;
};

vertex DisplayVertex display_vertex(uint vid [[vertex_id]],
                                    constant float2 &extent [[buffer(0)]]) {
    // Triangle strip: top-left, bottom-left, top-right, bottom-right.
    float2 corner = float2(vid & 2 ? 1.0 : -1.0, vid & 1 ? -1.0 : 1.0);
    float2 screen = float2(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);  // 0..1, y down
    DisplayVertex out;
    out.position = float4(corner * extent, 0.0, 1.0);
    out.uv = float2(screen.y, 1.0 - screen.x);  // 90 degrees clockwise
    return out;
}

fragment float4 display_fragment(DisplayVertex in [[stage_in]],
                                 texture2d<float> image [[texture(0)]],
                                 sampler linear [[sampler(0)]]) {
    float grey = image.sample(linear, in.uv).r;
    return float4(grey, grey, grey, 1.0);
}
