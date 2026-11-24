import time
from typing import Tuple, Optional
import gradio as gr
from PIL import Image

from src.pipeline import AtaaviVisionPipeline
from src.augmentations import RealWorldDegradations

# Initialize the inference pipeline once
print("Initializing Ataavi-Vision Pipeline...")
pipeline = AtaaviVisionPipeline(species_json_path="data/indian_birds.json")

def identify_bird(
    image: Optional[Image.Image],
    conf_thresh: float,
    pad_pct: float
) -> Tuple[Optional[Image.Image], Optional[Image.Image], dict, str]:
    if image is None:
        return None, None, {}, "Please upload an image to begin."

    output = pipeline.predict(
        image,
        conf_thresh=conf_thresh,
        pad_pct=pad_pct,
        top_k=3
    )

    annotated = output.annotated_image

    if len(output.detections) > 0:
        primary_det = output.detections[0]
        crop = primary_det["crop"]
        top_preds = primary_det["top_predictions"]
        
        # Build dictionary of predictions for gr.Label
        pred_dict = {
            f"{p.species} ({p.scientific_name})": float(p.confidence)
            for p in top_preds
        }

        fallback_msg = " [Fallback: Entire Frame Used]" if output.fallback_used else ""
        timing_info = (
            f"**Inference Summary:**\n"
            f"- **YOLOv8 Detection:** {output.det_ms:.1f} ms{fallback_msg}\n"
            f"- **BioCLIP Classification:** {output.cls_ms:.1f} ms\n"
            f"- **Total Latency:** {output.total_ms:.1f} ms\n"
            f"- **Top-1 Diagnosis:** {top_preds[0].species} (`{top_preds[0].scientific_name}`)"
        )
    else:
        crop = None
        pred_dict = {}
        timing_info = "No detections."

    return annotated, crop, pred_dict, timing_info

def test_robustness(
    image: Optional[Image.Image],
    degradation_type: str,
    conf_thresh: float,
    pad_pct: float
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image], dict, str]:
    if image is None:
        return None, None, None, {}, "Please upload an image to test robustness."

    # Apply selected degradation transform
    degraded_image = RealWorldDegradations.apply_degradation(image, degradation_type)

    # Run pipeline on degraded image
    output = pipeline.predict(
        degraded_image,
        conf_thresh=conf_thresh,
        pad_pct=pad_pct,
        top_k=3
    )

    annotated = output.annotated_image
    primary_det = output.detections[0]
    crop = primary_det["crop"]
    top_preds = primary_det["top_predictions"]

    pred_dict = {
        f"{p.species} ({p.scientific_name})": float(p.confidence)
        for p in top_preds
    }

    status_tag = "⚠️ Detector Missed / Full Fallback" if output.fallback_used else "✅ Avian Object Localized"
    report = (
        f"**Robustness Stress-Test Analysis:**\n"
        f"- **Degradation Mode:** `{degradation_type}`\n"
        f"- **Detection Status:** {status_tag}\n"
        f"- **Latency Breakdown:** Det: {output.det_ms:.1f} ms | Cls: {output.cls_ms:.1f} ms | Total: {output.total_ms:.1f} ms\n"
        f"- **Stress-Tested Prediction:** {top_preds[0].species} ({top_preds[0].confidence*100:.1f}%)"
    )

    return degraded_image, annotated, crop, pred_dict, report

custom_css = """
#header { text-align: center; margin-bottom: 1.5rem; }
.output-box { border-radius: 8px; }
"""

with gr.Blocks(title="Ataavi-Vision: Indian Bird Detection & Zero-Shot Classification", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🦜 Ataavi-Vision
        ### Production-Grade Indian Bird Detection & Zero-Shot Classification
        *Ultralytics YOLOv8 (Avian Localization & Padded Cropping) + Imageomics BioCLIP (Constrained Taxonomic Zero-Shot Inference)*
        """,
        elem_id="header"
    )

    with gr.Tabs():
        with gr.TabItem("🔍 Live Bird Identifier"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(type="pil", label="Upload Avian Photo")
                    with gr.Accordion("⚙️ Detector Tuning Parameters", open=True):
                        conf_slider = gr.Slider(
                            minimum=0.05, maximum=0.90, value=0.20, step=0.05,
                            label="YOLO Confidence Threshold"
                        )
                        pad_slider = gr.Slider(
                            minimum=0.0, maximum=0.40, value=0.15, step=0.05,
                            label="Bounding Box Margin Padding (%)"
                        )
                    run_btn = gr.Button("⚡ Run Two-Stage Identification", variant="primary")

                with gr.Column(scale=1):
                    annotated_out = gr.Image(type="pil", label="1. Localized Bird (Bounding Box)")
                    with gr.Row():
                        crop_out = gr.Image(type="pil", label="2. Padded Region Crop", scale=1)
                        label_out = gr.Label(num_top_classes=3, label="3. Top-3 Taxonomic Predictions", scale=2)
                    timing_out = gr.Markdown("Ready for upload.")

            run_btn.click(
                fn=identify_bird,
                inputs=[input_img, conf_slider, pad_slider],
                outputs=[annotated_out, crop_out, label_out, timing_out]
            )

        with gr.TabItem("🧪 Robustness Stress-Test Lab"):
            gr.Markdown(
                """
                Simulate difficult real-world field birding conditions to observe how the two-stage pipeline
                behaves under optical noise, underexposure, leaf occlusions, and long-range telephoto downscaling.
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    rob_input_img = gr.Image(type="pil", label="Source Field Photo")
                    deg_selector = gr.Radio(
                        choices=[
                            ("Low Light (Underexposure + ISO Shot Noise)", "low_light"),
                            ("Motion Blur (Handheld Camera Shake)", "motion_blur"),
                            ("Foliage Occlusion (25-35% Leaf Coverage)", "foliage_occlusion"),
                            ("Distant Bird (<10% Frame Area + Telephoto Bokeh)", "distant_bird")
                        ],
                        value="low_light",
                        label="Select Real-World Degradation Transform"
                    )
                    with gr.Row():
                        rob_conf_slider = gr.Slider(minimum=0.05, maximum=0.80, value=0.20, step=0.05, label="YOLO Conf")
                        rob_pad_slider = gr.Slider(minimum=0.0, maximum=0.30, value=0.15, step=0.05, label="Pad %")
                    rob_btn = gr.Button("🔬 Stress-Test Pipeline", variant="secondary")

                with gr.Column(scale=1):
                    with gr.Row():
                        rob_degraded_out = gr.Image(type="pil", label="Degraded Optical Frame")
                        rob_annotated_out = gr.Image(type="pil", label="Stress-Tested Detection")
                    with gr.Row():
                        rob_crop_out = gr.Image(type="pil", label="Crop Extracted", scale=1)
                        rob_label_out = gr.Label(num_top_classes=3, label="Predicted Species", scale=2)
                    rob_report_out = gr.Markdown("Select degradation mode to evaluate.")

            rob_btn.click(
                fn=test_robustness,
                inputs=[rob_input_img, deg_selector, rob_conf_slider, rob_pad_slider],
                outputs=[rob_degraded_out, rob_annotated_out, rob_crop_out, rob_label_out, rob_report_out]
            )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
