/**
 * ComfyUI web extension for the MSR Contact Sheet Cropper node pack.
 *
 * Adds distinctive node colors to the Licon-MSR / Crop and
 * Licon-MSR / Assemble categories so users can spot them quickly in
 * large workflows.
 */

import { app } from "../../scripts/app.js";

const MSR_NODE_STYLES = {
    MSRContactSheetCropper: {
        color: "#4a4a6a",
        bgcolor: "#2a2a40",
        groupcolor: "#6a6aff",
    },
    MSRContactSheetAssembler: {
        color: "#4a6a4a",
        bgcolor: "#2a402a",
        groupcolor: "#6aff6a",
    },
};

app.registerExtension({
    name: "ComfyUI.MSR.ContactSheet",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const styles = MSR_NODE_STYLES[nodeData.name];
        if (!styles) {
            return;
        }

        // Set default node styling. These values are used when the node
        // is created in the graph.
        nodeType.prototype.onNodeCreated = function () {
            this.color = styles.color;
            this.bgcolor = styles.bgcolor;
            this.groupcolor = styles.groupcolor;
        };
    },
});
