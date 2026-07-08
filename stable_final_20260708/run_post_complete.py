import os
import json
import shutil
import argparse
import hashlib
import time
from pathlib import Path
from dotenv import load_dotenv

from perception.yolo_world_engine import YoloWorldEngine
from perception.fastsam_engine import FastSAMEngine
from perception.depth_engine import DepthEngine
from planning.validator_engine import LogicValidator, PostCheckValidator

# Import Langfuse context to ensure background sync completes
from langfuse.decorators import langfuse_context

from capture_config import (
    USER_QUERY,
    PROJECT_DIR,
    ENV_PATH,
    BASE_DIR,
    TEST_NAME,
    STEP_INDEX,

    IMAGE_PATH,
    DEPTH_PATH,
    INTRINSICS_PATH,

    YOLO_WORLD_MODEL_PATH,
    FASTSAM_MODEL_PATH,

    POST_IMAGE_PATH,
    VALIDATION_JSON,
    POST_OUTPUT_DIR,
    POST_CHECK_JSON,
    POST_YOLO_JSON,
    POST_YOLO_IMAGE,
    POST_FASTSAM_MASK,
    POST_FASTSAM_IMAGE,
    REMAINING_PLAN_JSON,

    VISION_OUTPUT_DIR,
    SNAPSHOT_POST_RGB,
    print_config,
)


# ============================================================
# INIT
# ============================================================

load_dotenv(ENV_PATH)

PROJECT_DIR = Path(PROJECT_DIR)
VISION_OUTPUT_DIR = Path(VISION_OUTPUT_DIR)
POST_OUTPUT_DIR = Path(POST_OUTPUT_DIR)

VISION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
POST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# BASIC IO
# ============================================================

def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path, label="JSON"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")

    with open(path, "r") as f:
        return json.load(f)


def copy_if_exists(src, dst):
    src = Path(src)
    dst = Path(dst)

    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"[COPY] {src} -> {dst}")
        return True

    print(f"[WARN] File tidak ditemukan, skip copy: {src}")
    return False


def file_sha256(path, chunk_size=1024 * 1024):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_info(path):
    path = Path(path)
    if not path.exists():
        return {"exists": False, "path": str(path)}
    st = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": int(st.st_size),
        "mtime_unix": float(st.st_mtime),
        "mtime_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        "sha256": file_sha256(path),
    }


def sync_current_rgb_to_post_image():
    if not Path(IMAGE_PATH).exists():
        raise FileNotFoundError(f"IMAGE_PATH tidak ditemukan: {IMAGE_PATH}")
    if not Path(DEPTH_PATH).exists():
        raise FileNotFoundError(f"DEPTH_PATH tidak ditemukan: {DEPTH_PATH}")
    if not Path(INTRINSICS_PATH).exists():
        raise FileNotFoundError(f"INTRINSICS_PATH tidak ditemukan: {INTRINSICS_PATH}")

    ensure_parent(POST_IMAGE_PATH)
    shutil.copyfile(IMAGE_PATH, POST_IMAGE_PATH)

    sync = {
        "test_name": TEST_NAME,
        "step_index": STEP_INDEX,
        "current_rgb": file_info(IMAGE_PATH),
        "post_rgb": file_info(POST_IMAGE_PATH),
        "depth_raw": file_info(DEPTH_PATH),
        "intrinsics": file_info(INTRINSICS_PATH),
        "post_rgb_is_exact_copy_of_current_rgb": file_sha256(IMAGE_PATH) == file_sha256(POST_IMAGE_PATH),
        "note": "post_scene_rgb.jpg dibuat dengan copyfile dari current_scene_rgb.jpg terbaru sebelum post-check.",
    }

    sync_json = POST_OUTPUT_DIR / f"STEP_{STEP_INDEX}_sync_report.json"
    save_json(sync_json, sync)

    print("\n===== SYNC CHECK CURRENT/POST/DEPTH =====")
    print("current RGB:", IMAGE_PATH)
    print("post RGB   :", POST_IMAGE_PATH)
    print("depth raw  :", DEPTH_PATH)
    print("post == current:", sync["post_rgb_is_exact_copy_of_current_rgb"])
    print("sync report:", sync_json)

    if not sync["post_rgb_is_exact_copy_of_current_rgb"]:
        raise RuntimeError("post_scene_rgb.jpg tidak identik dengan current_scene_rgb.jpg")

    return sync


def safe_name(text):
    text = str(text).strip().lower()
    for ch in ["/", "\\", ":", ";", ",", ".", "(", ")", "[", "]", "{", "}", "'", '"']:
        text = text.replace(ch, "_")
    text = "_".join(text.split())
    return text


# ============================================================
# SNAPSHOT
# ============================================================

def save_post_snapshot():
    if os.path.exists(POST_IMAGE_PATH):
        ensure_parent(SNAPSHOT_POST_RGB)
        shutil.copyfile(POST_IMAGE_PATH, SNAPSHOT_POST_RGB)
        print("Saved post RGB snapshot:", SNAPSHOT_POST_RGB)
    else:
        print("[WARN] POST_IMAGE_PATH tidak ada, snapshot tidak disimpan:", POST_IMAGE_PATH)


# ============================================================
# TEXT NORMALIZATION + ALIAS
# ============================================================

def normalize_repeated_words(text):
    if text is None:
        return ""

    words = str(text).lower().strip().split()
    clean_words = []

    for word in words:
        if not clean_words or clean_words[-1] != word:
            clean_words.append(word)

    return " ".join(clean_words)


def build_yolo_aliases(target):
    target = normalize_repeated_words(target)
    aliases = [target]

    if "lotion" in target or "bottle" in target:
        aliases.extend(["pink lotion bottle", "lotion bottle", "body lotion", "pink bottle", "bottle", "lotion"])

    if "soda" in target or "can" in target:
        aliases.extend(["soda can", "red soda can", "can", "drink can", "beverage can"])

    if "lemon" in target:
        aliases.extend(["lemon", "yellow lemon", "fruit"])
    if "orange" in target:
        aliases.extend(["orange", "orange fruit", "fruit"])
    if "apple" in target:
        aliases.extend(["apple", "red apple", "fruit"])
    if "bell pepper" in target or "pepper" in target:
        aliases.extend(["bell pepper", "pepper", "vegetable"])

    unique = []
    for alias in aliases:
        alias = normalize_repeated_words(alias)
        if alias and alias not in unique:
            unique.append(alias)

    return unique


# ============================================================
# VALIDATION RESULT
# ============================================================

def get_provider_keys():
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    # ollama_key = os.getenv("OLLAMA_")
    return openai_key, gemini_key

def create_validator(openai_key, gemini_key):
    use_local = os.getenv("USE_LOCAL_VLM", "False").lower() in ("true", "1", "yes")
    if use_local:
        return LogicValidator(api_key=None, provider="local")
    if openai_key:
        return LogicValidator(api_key=openai_key, provider="openai")
    if gemini_key:
        return LogicValidator(api_key=gemini_key, provider="gemini")
    raise RuntimeError("Tidak ada API key untuk validator.")


def create_postcheck_validator():

    use_local = os.getenv(
        "USE_LOCAL_VLM",
        "False"
    ).lower() in ("true","1","yes")

    if use_local:
        return PostCheckValidator(
            api_key=None,
            provider="local"
        )

    openai_key, gemini_key = get_provider_keys()

    if openai_key:
        return PostCheckValidator(
            api_key=openai_key,
            provider="openai"
        )

    if gemini_key:
        return PostCheckValidator(
            api_key=gemini_key,
            provider="gemini"
        )

    raise RuntimeError("No validator available")


def vlm_postcheck(target):

    validator=create_postcheck_validator()
    response=validator.validate_target_presence(
            image_path=POST_IMAGE_PATH,
            target=target
    )

    found=response.get(
            "target_found_after_action",
            False
    )

    confidence=response.get(
            "confidence",
            0.0
    )

    if found:
        best_detection={
            "label":target,
            "confidence":confidence,
            "source":"vlm_postcheck",
            "target_query":target
        }
        detections=[best_detection]

    else:
        best_detection=None
        detections=[]


    return found,best_detection,detections

def revalidate_remaining_plan(remaining_plan):
    """
    Melakukan reasoning ulang apakah sisa action_plan masih logis 
    untuk dieksekusi berdasarkan kondisi lingkungan terbaru setelah aksi sebelumnya.
    """
    if not remaining_plan:
        return True, "Tidak ada step tersisa, tidak perlu validasi."

    print("\n🔍 MELAKUKAN REASONING ULANG TERHADAP SISA PLAN...")
    openai_key, gemini_key = get_provider_keys()
    # validator = create_validator(openai_key, gemini_key)
    validator = LogicValidator(provider="local", model_name="qwen3.5:27b")

    temp_planner_json = {
        "action_plan": remaining_plan
    }

    validation_result = validator.validate_strategy(
        image_path=POST_IMAGE_PATH,
        user_query=USER_QUERY,
        planner_json=temp_planner_json
    )

    print(json.dumps(validation_result, indent=2))

    status = str(validation_result.get("validation_status", "FAIL")).upper()
    feedback = validation_result.get("feedback", "No feedback provided.")

    print(f"Status Re-validasi: {status}")
    print(f"Feedback VLM: {feedback}")

    is_valid = (status == "PASS")
    
    revalidation_log = POST_OUTPUT_DIR / f"STEP_{STEP_INDEX}_revalidation_report.json"
    save_json(revalidation_log, validation_result)

    return is_valid, feedback

def load_validation_result(validation_path):
    return load_json(validation_path, label="Validation JSON")


def get_target_step_from_validation(validation_result, step_index):
    status = str(validation_result.get("validation_status", "FAIL")).upper()

    if status != "PASS":
        raise RuntimeError("Plan belum PASS. Tidak boleh lanjut ke post-check.")

    final_plan = validation_result.get("final_action_plan", [])

    if not isinstance(final_plan, list) or len(final_plan) == 0:
        raise RuntimeError("final_action_plan kosong.")

    if not isinstance(step_index, int):
        raise RuntimeError(f"STEP_INDEX harus integer, dapat: {step_index}")

    if step_index < 1:
        raise RuntimeError(f"STEP_INDEX harus mulai dari 1, dapat: {step_index}")

    list_index = step_index - 1

    if list_index >= len(final_plan):
        raise RuntimeError(f"STEP_INDEX={step_index} melebihi jumlah action_plan. Jumlah step tersedia: {len(final_plan)}")

    selected_step = final_plan[list_index]

    target = selected_step.get("target")
    target = normalize_repeated_words(target)

    if not target:
        raise RuntimeError(f"Target pada STEP_INDEX={step_index} kosong.")

    print(f"\nSTEP_INDEX dari config: {step_index}")
    print("Target untuk post-check:", target)

    return target, selected_step, final_plan


# ============================================================
# POST CHECK
# ============================================================

def run_post_check(target):

    mode=os.getenv("POSTCHECK_MODE","yolo")

    if not os.path.exists(POST_IMAGE_PATH):
        raise FileNotFoundError(POST_IMAGE_PATH)

    if mode=="vlm":
        target_found,_,_=vlm_postcheck(target)
        print("validasi by vlm")

        if not target_found:
            result={
                "test_name":TEST_NAME,
                "step_index":STEP_INDEX,
                "target":target,
                "post_check_status":"REMOVED_SUCCESS",
                "target_found_after_action":False,
                "best_detection":None,
                "all_detections":[],
                "post_image_path":POST_IMAGE_PATH,
                "post_yolo_image":None,
                "post_fastsam_mask":None,
                "post_fastsam_image":None,
                "note":"validated by VLM"
            }

            save_json(POST_CHECK_JSON,result)
            return result
        
        print("VLM says object still exists")
        print("Switching to YOLO localization")

    elif mode=="yolo":
        print("Pure YOLO post-check")

    yolo = YoloWorldEngine(
        model_name=YOLO_WORLD_MODEL_PATH,
        conf=0.5,
        output_dir=str(POST_OUTPUT_DIR),
    )

    try:
        best_detection,all_detections=yolo.detect_target(
            image_path=POST_IMAGE_PATH,
            target=target,
            output_json=POST_YOLO_JSON,
            output_image=POST_YOLO_IMAGE,
            conf=0.5,
            use_generic_fallback=False
        )

        bbox=best_detection["bbox"]
        fastsam=FastSAMEngine(
            model_name=FASTSAM_MODEL_PATH,
            device="cpu",
            imgsz=640,
            conf=0.4,
            iou=0.9,
            output_dir=str(POST_OUTPUT_DIR),
        )

        post_fastsam_mask,post_fastsam_image= fastsam.segment_bbox(
                image_path=POST_IMAGE_PATH,
                bbox=bbox,
                mask_path=POST_FASTSAM_MASK,
                result_image_path=POST_FASTSAM_IMAGE,
            )

        result={
            "test_name":TEST_NAME,
            "step_index":STEP_INDEX,
            "target":target,
            "post_check_status":"STILL_FOUND",
            "target_found_after_action":True,
            "best_detection":best_detection,
            "all_detections":all_detections,
            "post_image_path":POST_IMAGE_PATH,
            "post_yolo_image":POST_YOLO_IMAGE,
            "post_fastsam_mask":post_fastsam_mask,
            "post_fastsam_image":post_fastsam_image,
            "note":"validated by VLM + localized by YOLO"
        }
    except RuntimeError:
        result={
            "test_name":TEST_NAME,

            "step_index":STEP_INDEX,

            "target":target,
            "post_check_status":"REMOVED_SUCCESS",
            "target_found_after_action":False,
            "best_detection":None,
            "all_detections":[],
            "post_image_path":POST_IMAGE_PATH,
            "post_yolo_image":None,
            "post_fastsam_mask":None,
            "post_fastsam_image":None,
            "note":"VLM found target but YOLO failed"
        }

    save_json(POST_CHECK_JSON,result)

    return result



# def run_post_check(target):
#     mode=os.getenv(
#             "POSTCHECK_MODE",
#             "yolo"
#     )
#     if not os.path.exists(POST_IMAGE_PATH):
#         raise FileNotFoundError(
#             f"Gambar post-check tidak ditemukan: {POST_IMAGE_PATH}\n"
#             f"Capture scene terbaru setelah aksi robot, lalu copy current_scene_rgb.jpg menjadi post_scene_rgb.jpg."
#         )

#     yolo = YoloWorldEngine(
#         model_name=YOLO_WORLD_MODEL_PATH,
#         conf=0.5,
#         output_dir=str(POST_OUTPUT_DIR),
#     )

#     try:
#         best_detection, all_detections = yolo.detect_target(
#             image_path=POST_IMAGE_PATH,
#             target=target,
#             output_json=POST_YOLO_JSON,
#             output_image=POST_YOLO_IMAGE,
#             conf=0.5,
#             use_generic_fallback=False,
#         )

#         target_found = True
#         post_status = "STILL_FOUND"

#         bbox = best_detection["bbox"]

#         fastsam = FastSAMEngine(
#             model_name=FASTSAM_MODEL_PATH,
#             device="cpu",
#             imgsz=640,
#             conf=0.4,
#             iou=0.9,
#             output_dir=str(POST_OUTPUT_DIR),
#         )

#         post_fastsam_mask, post_fastsam_image = fastsam.segment_bbox(
#             image_path=POST_IMAGE_PATH,
#             bbox=bbox,
#             mask_path=POST_FASTSAM_MASK,
#             result_image_path=POST_FASTSAM_IMAGE,
#         )

#         result = {
#             "test_name": TEST_NAME,
#             "step_index": STEP_INDEX,
#             "target": target,
#             "post_check_status": post_status,
#             "target_found_after_action": target_found,
#             "best_detection": best_detection,
#             "all_detections": all_detections,
#             "post_image_path": POST_IMAGE_PATH,
#             "post_yolo_image": POST_YOLO_IMAGE,
#             "post_fastsam_mask": post_fastsam_mask,
#             "post_fastsam_image": post_fastsam_image,
#             "note": "Target masih terdeteksi setelah aksi robot. Step belum dianggap selesai.",
#         }

#     except RuntimeError as e:
#         error_msg = str(e).lower()

#         if "tidak menemukan target" in error_msg or "not found" in error_msg:
#             result = {
#                 "test_name": TEST_NAME,
#                 "step_index": STEP_INDEX,
#                 "target": target,
#                 "post_check_status": "REMOVED_SUCCESS",
#                 "target_found_after_action": False,
#                 "best_detection": None,
#                 "all_detections": [],
#                 "post_image_path": POST_IMAGE_PATH,
#                 "post_yolo_image": POST_YOLO_IMAGE,
#                 "post_fastsam_mask": None,
#                 "post_fastsam_image": None,
#                 "note": "Target tidak ditemukan setelah aksi robot. Step dianggap berhasil.",
#             }
#         else:
#             raise

#     save_json(POST_CHECK_JSON, result)

#     print("\nPost-check selesai.")
#     print(f"Saved to: {POST_CHECK_JSON}")
#     print(json.dumps(result, indent=2, ensure_ascii=False))

#     return result


# ============================================================
# REMAINING PLAN
# ============================================================

def create_remaining_plan_after_success(validation_result, post_check_result, step_index):
    final_plan = validation_result.get("final_action_plan", [])

    if not isinstance(final_plan, list):
        final_plan = []

    if post_check_result.get("post_check_status") != "REMOVED_SUCCESS":
        print("\nStep belum sukses. remaining_plan tidak diubah.")
        return final_plan

    if step_index < 1:
        raise RuntimeError(f"STEP_INDEX tidak valid: {step_index}")

    remaining_plan = final_plan[step_index:]
    save_json(REMAINING_PLAN_JSON, remaining_plan)

    print("\nRemaining plan saved to:", REMAINING_PLAN_JSON)
    print(json.dumps(remaining_plan))

    return remaining_plan


# ============================================================
# SAFETY / SANITY CHECK
# ============================================================

def sanity_check_object_position(object_position):
    point = object_position.get("point_camera_m")

    if not point or len(point) != 3:
        print("[WARN] object_position tidak punya point_camera_m valid.")
        return {"ok": False, "reason": "point_camera_m missing or invalid"}

    x, y, z = point
    warnings = []

    if z < 0.10: warnings.append("depth z terlalu dekat dari kamera")
    if z > 1.20: warnings.append("depth z terlalu jauh dari kamera")
    if abs(x) > 0.50: warnings.append("camera x terlalu besar")
    if abs(y) > 0.50: warnings.append("camera y terlalu besar")

    ok = len(warnings) == 0

    if ok:
        print("[OK] object_position camera sanity check masuk akal:", point)
    else:
        print("[WARN] object_position camera terlihat mencurigakan:", point)
        for w in warnings:
            print(" -", w)

    return {"ok": ok, "point_camera_m": point, "warnings": warnings}


# ============================================================
# YOLO NEXT TARGET WITH ALIAS
# ============================================================

def detect_next_target_with_aliases(yolo, image_path, next_target, next_output_dir, next_step_number, next_yolo_json, next_yolo_image):
    aliases = build_yolo_aliases(next_target)

    print("\n[YOLO-World] Trying aliases for next target:")
    for alias in aliases:
        print(" -", alias)

    conf_list = [0.50, 0.40, 0.30, 0.20]
    best_detection = None
    all_detections = []
    used_query = None
    used_conf = None
    last_error = None
    debug_results = []

    for conf in conf_list:
        for query in aliases:
            alias_safe = safe_name(query)
            alias_yolo_json = next_output_dir / f"STEP_{next_step_number}_next_yolo_{alias_safe}_conf_{str(conf).replace('.', '_')}.json"
            alias_yolo_image = next_output_dir / f"STEP_{next_step_number}_next_yolo_{alias_safe}_conf_{str(conf).replace('.', '_')}.jpg"

            print(f"\n[YOLO-World] Trying query='{query}' conf={conf}")

            try:
                best_detection, all_detections = yolo.detect_target(
                    image_path=image_path,
                    target=query,
                    output_json=str(alias_yolo_json),
                    output_image=str(alias_yolo_image),
                    conf=conf,
                    use_generic_fallback=True,
                )

                used_query = query
                used_conf = conf

                best_detection["target_query_original"] = next_target
                best_detection["target_query_used"] = used_query
                best_detection["confidence_threshold_used"] = used_conf

                copy_if_exists(alias_yolo_json, next_yolo_json)
                copy_if_exists(alias_yolo_image, next_yolo_image)

                debug_results.append({
                    "query": query, "conf": conf, "status": "FOUND",
                    "best_detection": best_detection, "output_json": str(alias_yolo_json), "output_image": str(alias_yolo_image)
                })

                print("[YOLO-World] SUCCESS")
                print("Used query:", used_query)
                print("Used conf:", used_conf)
                print(json.dumps(best_detection, indent=2, ensure_ascii=False))

                debug_json = next_output_dir / f"STEP_{next_step_number}_next_yolo_alias_debug.json"
                save_json(debug_json, {"target_original": next_target, "aliases": aliases, "conf_list": conf_list, "used_query": used_query, "used_conf": used_conf, "debug_results": debug_results})

                return best_detection, all_detections, used_query, used_conf

            except RuntimeError as e:
                last_error = str(e)
                debug_results.append({"query": query, "conf": conf, "status": "FAILED", "error": str(e), "output_json": str(alias_yolo_json), "output_image": str(alias_yolo_image)})
                print(f"[YOLO-World] Failed query='{query}' conf={conf}: {e}")

    debug_json = next_output_dir / f"STEP_{next_step_number}_next_yolo_alias_debug_FAILED.json"
    save_json(debug_json, {"target_original": next_target, "aliases": aliases, "conf_list": conf_list, "last_error": last_error, "debug_results": debug_results})

    raise RuntimeError(f"YOLO-World tidak menemukan target next step: {next_target}. Aliases tried: {aliases}. Conf tried: {conf_list}. Last error: {last_error}.")


# ============================================================
# NEXT TARGET PIPELINE
# ============================================================

def copy_next_target_outputs_to_main_vision(next_result):
    mappings = [
        (next_result["next_yolo_json"], VISION_OUTPUT_DIR / "detections_yolo.json"),
        (next_result["next_yolo_image"], VISION_OUTPUT_DIR / "yolo_world_result.jpg"),
        (next_result["next_fastsam_mask"], VISION_OUTPUT_DIR / "fastsam_mask.png"),
        (next_result["next_fastsam_image"], VISION_OUTPUT_DIR / "fastsam_result.jpg"),
        (next_result["next_object_position_json"], VISION_OUTPUT_DIR / "object_position_camera.json"),
    ]
    for src, dst in mappings:
        copy_if_exists(src, dst)


def process_next_target_after_success(remaining_plan):
    if not isinstance(remaining_plan, list) or len(remaining_plan) == 0:
        print("\n✅ Tidak ada target berikutnya untuk diproses.")
        return None

    if not os.path.exists(IMAGE_PATH): raise FileNotFoundError(f"IMAGE_PATH tidak ditemukan: {IMAGE_PATH}")
    if not os.path.exists(DEPTH_PATH): raise FileNotFoundError(f"DEPTH_PATH tidak ditemukan: {DEPTH_PATH}")
    if not os.path.exists(INTRINSICS_PATH): raise FileNotFoundError(f"INTRINSICS_PATH tidak ditemukan: {INTRINSICS_PATH}")

    next_step = remaining_plan[0]
    next_target = normalize_repeated_words(next_step.get("target", ""))

    if not next_target:
        print("\n⚠️ Target berikutnya kosong. Tidak bisa lanjut YOLO/FastSAM.")
        return None

    next_step_number = next_step.get("step", STEP_INDEX + 1)

    print("\n===== PROCESS NEXT TARGET AFTER POST-CHECK SUCCESS =====")
    print("Next step:")
    print(json.dumps(next_step, indent=2, ensure_ascii=False))
    print("Next target untuk YOLO/FastSAM:", next_target)

    next_output_dir = POST_OUTPUT_DIR / f"STEP_{next_step_number}_next_target"
    next_output_dir.mkdir(parents=True, exist_ok=True)

    next_yolo_json = next_output_dir / f"STEP_{next_step_number}_next_detections_yolo.json"
    next_yolo_image = next_output_dir / f"STEP_{next_step_number}_next_yolo_result.jpg"
    next_fastsam_mask = next_output_dir / f"STEP_{next_step_number}_next_fastsam_mask.png"
    next_fastsam_image = next_output_dir / f"STEP_{next_step_number}_next_fastsam_result.jpg"
    next_object_position_json = next_output_dir / f"STEP_{next_step_number}_next_object_position_camera.json"

    yolo = YoloWorldEngine(model_name=YOLO_WORLD_MODEL_PATH, conf=0.5, output_dir=str(next_output_dir))

    best_detection, all_detections, used_query, used_conf = detect_next_target_with_aliases(
        yolo=yolo, image_path=IMAGE_PATH, next_target=next_target, next_output_dir=next_output_dir,
        next_step_number=next_step_number, next_yolo_json=next_yolo_json, next_yolo_image=next_yolo_image,
    )

    fastsam = FastSAMEngine(model_name=FASTSAM_MODEL_PATH, device="cpu", imgsz=640, conf=0.4, iou=0.9, output_dir=str(next_output_dir))

    mask_path, fastsam_result_path = fastsam.segment_bbox(
        image_path=IMAGE_PATH, bbox=best_detection["bbox"], mask_path=str(next_fastsam_mask), result_image_path=str(next_fastsam_image),
    )

    depth = DepthEngine(depth_path=DEPTH_PATH, intrinsics_path=INTRINSICS_PATH, min_depth=0.1, max_depth=2.0)

    object_position = depth.extract_from_mask(
        target=next_target, mask_path=mask_path, output_path=str(next_object_position_json),
    )

    sanity = sanity_check_object_position(object_position)

    next_result = {
        "test_name": TEST_NAME, "current_step_index_completed": STEP_INDEX, "next_step": next_step,
        "next_target": next_target, "target_query_used": used_query, "confidence_threshold_used": used_conf,
        "best_detection": best_detection, "all_detections": all_detections, "next_rgb_image_used": IMAGE_PATH,
        "next_depth_used": DEPTH_PATH, "next_yolo_json": str(next_yolo_json), "next_yolo_image": str(next_yolo_image),
        "next_fastsam_mask": str(next_fastsam_mask), "next_fastsam_image": str(next_fastsam_image),
        "next_object_position_json": str(next_object_position_json), "object_position": object_position,
        "sanity_check_object_position": sanity,
        "note": "Next target has been detected, segmented, and localized.",
    }

    next_result_json = next_output_dir / f"STEP_{next_step_number}_next_target_result.json"
    save_json(next_result_json, next_result)

    print("\n✅ Next target pipeline selesai.")
    copy_next_target_outputs_to_main_vision(next_result)

    pipeline_ready_json = VISION_OUTPUT_DIR / "next_target_ready.json"
    save_json(pipeline_ready_json, {
        "ready": True, "test_name": TEST_NAME, "completed_step_index": STEP_INDEX, "next_step_number": next_step_number,
        "next_target": next_target, "vision_output_dir": str(VISION_OUTPUT_DIR),
        "standard_outputs": {
            "detections_yolo": str(VISION_OUTPUT_DIR / "detections_yolo.json"),
            "yolo_world_result": str(VISION_OUTPUT_DIR / "yolo_world_result.jpg"),
            "fastsam_mask": str(VISION_OUTPUT_DIR / "fastsam_mask.png"),
        },
    })

    return next_result


# ============================================================
# MAIN
# ============================================================

def main():
    print_config()
    sync_current_rgb_to_post_image()
    save_post_snapshot()

    validation_result = load_validation_result(VALIDATION_JSON)
    target, selected_step, final_plan = get_target_step_from_validation(validation_result, STEP_INDEX)

    post_check_result = run_post_check(target)
    remaining_plan = create_remaining_plan_after_success(validation_result, post_check_result, STEP_INDEX)

    if post_check_result.get("post_check_status") == "REMOVED_SUCCESS":
        print("\n✅ Step yang diverifikasi berhasil menurut post-check.")

        if len(remaining_plan) > 0:
            print("➡️ Masih ada step berikutnya.")
            
            is_still_valid, vlm_feedback = revalidate_remaining_plan(remaining_plan)

            if not is_still_valid:
                print("\n🛑 STOP! LINGKUNGAN TELAH BERUBAH SECARA DRASTIS.")
                print(f"Alasan VLM: {vlm_feedback}")
                print("Sisa plan sudah tidak aman atau tidak relevan.")
                print("TINDAKAN: Kembali ke script run_d455_pipeline.py untuk RE-PLANNING total.")
                
                # Simpan marker untuk sistem bash script agar tahu harus re-plan
                save_json(VISION_OUTPUT_DIR / "require_replanning.json", {"require_replan": True, "reason": vlm_feedback})
                return # Keluar dari script, jangan lanjutkan proses next target
            # -----------------------------------

            print("\n✅ VLM menyatakan sisa plan masih valid dan aman. Melanjutkan proses...")
            print("Target berikutnya:", remaining_plan[0].get("target"))

            print("\n🔍 Memproses YOLO/FastSAM/depth untuk target berikutnya...")
            next_result = process_next_target_after_success(remaining_plan)
        else:
            print("✅ Semua step dalam action_plan sudah selesai.")
    else:
        print("\n⚠️ Step yang diverifikasi belum berhasil.")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensures Langfuse background threads finish uploading before script exit
        langfuse_context.flush()