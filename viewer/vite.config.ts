import { defineConfig } from "vite";
import { mkdir, writeFile, readFile } from "node:fs/promises";
import {createHash} from "node:crypto";
import { resolve } from "node:path";
export default defineConfig({
  server: {proxy: {"/__robotics": {target: "ws://127.0.0.1:8767", ws: true},"/__realism": {target: "ws://127.0.0.1:8766", ws: true}}},
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        forest: resolve(import.meta.dirname, "forest.html"),
        robotics: resolve(import.meta.dirname, "robotics.html"),
        realism: resolve(import.meta.dirname, "realism.html"),
        mapping: resolve(import.meta.dirname, "mapping.html"),
        fleet: resolve(import.meta.dirname, "fleet.html"),
      },
    },
  },
  plugins: [
    {
      name: "local-forest-capture",
      configureServer(server) {
        server.middlewares.use("/__forest-capture", async (req, res) => {
          if (req.method !== "POST") {
            res.statusCode = 405;
            res.end("POST required");
            return;
          }
          try {
            let body = "";
            for await (const chunk of req) {
              body += chunk;
              if (body.length > 24000000) throw Error("Capture too large");
            }
            const payload = JSON.parse(body);
            if (
              typeof payload.rgb !== "string" ||
              !payload.rgb.startsWith("data:image/png;base64,")
            )
              throw Error("PNG capture required");
            const folder = resolve(
              import.meta.dirname,
              "../artifacts/forest-quality-20260905",
            );
            await mkdir(folder, { recursive: true });
            const name = `${payload.view === "observer" ? "observer" : "camera"}-${Math.round(Number(payload.time_s) * 10)}`;
            const { rgb, ...metadata } = payload;
            const hash=createHash("sha256");
            for(const file of ["main.ts","trees.ts","textures.ts","ground.ts","understory.ts","cabin.ts","sunrise.ts"]) hash.update(await readFile(resolve(import.meta.dirname,"src/forest",file)));
            metadata.renderer_source_sha256=hash.digest("hex");
            await writeFile(
              resolve(folder, name + ".png"),
              Buffer.from(rgb.split(",")[1], "base64"),
            );
            await writeFile(
              resolve(folder, name + ".json"),
              JSON.stringify(metadata, null, 2),
            );
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ file: name + ".png" }));
          } catch (e) {
            res.statusCode = 400;
            res.end(String(e));
          }
        });
      },
    },
  ],
});
