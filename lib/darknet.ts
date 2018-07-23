import * as ffi from 'ffi';
import * as ref from 'ref';
import * as Struct from 'ref-struct';
import {readFileSync} from 'fs';

const debug = require('debug')('darknet');

const char_pointer = ref.refType('char');
const float_pointer = ref.refType('float');
const float_pointer_pointer = ref.coerceType('float **');
const int_pointer = ref.refType('int');

const BBOX = Struct({
    'x': 'float',
    'y': 'float',
    'w': 'float',
    'h': 'float'
});

const DETECTION = Struct({
    'bbox': BBOX,
    'classes': 'int',
    'prob': float_pointer,
    'mask': float_pointer,
    'objectness': 'float',
    'sort_class': 'int'
});

const IMAGE = Struct({
    'w': 'int',
    'h': 'int',
    'c': 'int',
    'data': float_pointer
});

const METADATA = Struct({
    'classes': 'int',
    'names': 'string'
});

const detection_pointer = ref.refType(DETECTION);

const library = __dirname + "/libdarknet";

export class DarknetBase {

    darknet: any;
    meta: any;
    net: any;
    netSize: number;

    names: string[];
    private memoryIndex: number = 0;
    private memorySlotsUsed: number = 0;
    private memoryCount: number;
    private memory: any;

    /**
     * A new instance of pjreddie's darknet. Create an instance as soon as possible in your app, because it takes a while to init.
     * @param config
     */
    constructor(config: IDarknetConfig) {

        if (!config) throw new Error("A config file is required");
        if (!config.names && !config.namefile) throw new Error("Config must include detection class names");
        if (!config.names && config.namefile) config.names = readFileSync(config.namefile, 'utf8').split('\n');
        if (!config.names) throw new Error("No names detected.");
        if (!config.config) throw new Error("Config must include location to yolo config file");
        if (!config.weights) throw new Error("config must include the path to trained weights");

        this.names = config.names.filter(a => a.split("").length > 0);
        this.memoryCount = config.memory || 3;

        this.meta = new METADATA;
        this.meta.classes = this.names.length;
        this.meta.names = this.names.join('\n');

        this.darknet = ffi.Library(library, {
            'float_to_image': [IMAGE, ['int', 'int', 'int', float_pointer]],
            'load_image_color': [IMAGE, ['string', 'int', 'int']],
            'network_predict_image': [float_pointer, ['pointer', IMAGE]],
            'get_network_boxes': [detection_pointer, ['pointer', 'int', 'int', 'float', 'float', int_pointer, 'int', int_pointer]],
            'do_nms_obj': ['void', [detection_pointer, 'int', 'int', 'float']],
            'free_image': ['void', [IMAGE]],
            'free_detections': ['void', [detection_pointer, 'int']],
            'load_network': ['pointer', ['string', 'string', 'int']],
            'get_metadata': [METADATA, ['string']],

            'network_output_size': ['int', ['pointer']],
            'network_remember_memory': ['void', ['pointer', float_pointer_pointer, 'int']],
            'network_avg_predictions': [detection_pointer, ['pointer', 'int', float_pointer_pointer, 'int', int_pointer, 'int', 'int', 'float', 'float']],

            'network_memory_make': [float_pointer_pointer, ['int', 'int']],
            'network_memory_free': ['void', [float_pointer_pointer, 'int']]
        });

        this.net = this.darknet.load_network(config.config, config.weights, 0);

        this.netSize = this.darknet.network_output_size(this.net);

        this.makeMemory();
    }

    resetMemory({memory = this.memoryCount}: { memory?: number } = {}) {
        debug(`resetting memory to ${memory} slots`);
        this.darknet.network_memory_free(this.memory, this.memoryCount);

        this.memoryCount = memory;
        this.makeMemory();
    }

    private makeMemory() {
        debug('making memory');
        this.memoryIndex = 0;
        this.memory = this.darknet.network_memory_make(this.memoryCount, this.netSize);

        this.memorySlotsUsed = 0;
    }

    private async rememberNet() {

        debug('remember net', {index: this.memoryIndex, slots: this.memorySlotsUsed});

        await new Promise((res, rej) => this.darknet.network_remember_memory.async(
            this.net,
            this.memory,
            this.memoryIndex,
            (e: any) => e ? rej(e) : res())
        );

        this.memoryIndex = (this.memoryIndex + 1) % this.memoryCount;
        this.memorySlotsUsed = Math.min(this.memorySlotsUsed + 1, this.memoryCount);
    }

    private async avgPrediction({w, h, thresh, hier}: { w: number, h: number, thresh: number, hier: number }): Promise<{ buff: Buffer, num: number }> {

        debug('predicting');
        let pnum = ref.alloc('int');

        const buff = await new Promise<Buffer>((res, rej) => this.darknet.network_avg_predictions.async(
            this.net,
            this.netSize,
            this.memory,
            this.memorySlotsUsed,
            pnum,
            w, h,
            thresh, hier,
            (e: any, d: Buffer) => e ? rej(e) : res(d)
        ));

        return {buff, num: (pnum as any).deref()};
    }

    private getArrayFromBuffer(buffer: Buffer, length: number, type: ref.Type): number[] {
        let array = [];
        for (let i = 0; i < length; i++) {
            array.push(ref.get(ref.reinterpret(buffer, type.size, i * type.size), 0, type));
        }
        return array;
    }

    private bufferToDetections(buffer: Buffer, length: number): Detection[] {
        let detections: Detection[] = [];
        for (let i = 0; i < length; i++) {
            let det = ref.get(ref.reinterpret(buffer, 48, i * DETECTION.size), 0, DETECTION);
            let prob = this.getArrayFromBuffer(det.prob, this.meta.classes, ref.types.float);

            for (let j = 0; j < this.meta.classes; j++) {
                if (prob[j] > 0) {
                    let b = det.bbox;
                    detections.push({
                        name: this.names[j],
                        prob: prob[j],
                        box: {
                            x: b.x,
                            y: b.y,
                            w: b.w,
                            h: b.h
                        }
                    });
                }
            }
        }
        return detections;
    }

    predictionBufferToDetections(buffer: Buffer, length: number): Detection[] {
        return this.bufferToDetections(buffer, length);
    }

    protected async _detectAsync(net: any, meta: any, image: any, thresh?: number, hier_thresh?: number, nms?: number): Promise<Detection[]> {

        await new Promise((res, rej) =>
            this.darknet.network_predict_image.async(net, image, (e: any) => e ? rej(e) : res())
        );

        await this.rememberNet();

        const {buff: dets, num} = await this.avgPrediction({
            w: image.w as number,
            h: image.h as number,
            hier: hier_thresh as number,
            thresh: thresh as number
        });

        debug('doing nms');
        await new Promise((res, rej) =>
            this.darknet.do_nms_obj.async(
                dets, num, meta.classes, nms,
                (e: any) => e ? rej(e) : res()
            )
        );

        debug(`interpreting ${num} result`);
        const detections = this.bufferToDetections(dets, num);
        debug('finalising');
        this.darknet.free_detections(dets, num);
        return detections;
    }

    /**
     * Get a Darknet Image async from path
     * @param path
     * @returns Promise<IMAGE>
     */
    async getImageFromPathAsync(path: String) {
        return new Promise((res, rej) =>
            this.darknet.load_image_color.async(
                path, 0, 0,
                (e: any, image: any) => e ? rej(e) : res(image)
            )
        );
    }

    /**
     * convert darknet image to rgb buffer
     * @param {IMAGE} image
     * @returns {Buffer}
     */
    imageToRGBBuffer(image: any) {
        const w = image.w;
        const h = image.h;
        const c = image.c;

        const imageElements = w * h * c;

        const imageData = new Float32Array(
            image.data.reinterpret(imageElements * Float32Array.BYTES_PER_ELEMENT, 0).buffer,
            0,
            imageElements
        );

        const rgbBuffer = Buffer.allocUnsafe(imageData.length);

        const step = c * w;
        let i, k, j;

        for (i = 0; i < h; ++i) {
            for (k = 0; k < c; ++k) {
                for (j = 0; j < w; ++j) {
                    rgbBuffer[i * step + j * c + k] = imageData[k * w * h + i * w + j] * 255;
                }
            }
        }

        return rgbBuffer;
    }

    private rgbToDarknet(buffer: Buffer, w: number, h: number, c: number): Float32Array {
        const imageElements = w * h * c;
        const floatBuff = new Float32Array(imageElements);
        const step = w * c;

        let i, k, j;

        for (i = 0; i < h; ++i) {
            for (k = 0; k < c; ++k) {
                for (j = 0; j < w; ++j) {
                    floatBuff[k * w * h + i * w + j] = buffer[i * step + j * c + k] / 255;
                }
            }
        }

        return floatBuff;
    }

    getImageFromDarknetBuffer(buffer: Float32Array, w: number, h: number, c: number) {
        return this.darknet.float_to_image(
            w, h, c,
            new Uint8Array(
                buffer.buffer,
                0,
                buffer.length * Float32Array.BYTES_PER_ELEMENT
            )
        );
    }

    /**
     * Transform an RGB buffer to a darknet encoded image
     * @param buffer - rgb buffer
     * @param w - width
     * @param h - height
     * @param c - channels
     * @returns {Promise<IMAGE>}
     */
    async RGBBufferToImageAsync(buffer: Buffer, w: number, h: number, c: number): Promise<any> {
        const floatBuff = this.rgbToDarknet(buffer, w, h, c);

        return new Promise((res, rej) => this.darknet.float_to_image.async(
            w, h, c,
            new Uint8Array(
                floatBuff.buffer,
                0,
                floatBuff.length * Float32Array.BYTES_PER_ELEMENT
            ),
            (e: any, image: any) => e ? rej(e) : res(image)
        ));
    }

    /**
     * Asynchronously detect objects in an image.
     * @param image
     * @param config
     * @returns A promise
     */
    async detectAsync(image: string | IBufferImage, config?: IConfig): Promise<Detection[]> {
        if (!config) config = {};
        let thresh = (config.thresh) ? config.thresh : 0.5;
        let hier_thresh = (config.hier_thresh) ? config.hier_thresh : 0.5;
        let nms = (config.nms) ? config.nms : 0.5;

        const darkNetLoadedImage = typeof image === 'string';

        const imageData = typeof image === 'string' ?
            await this.getImageFromPathAsync(image) :
            await this.RGBBufferToImageAsync(image.b, image.w, image.h, image.c);

        const detection = await this._detectAsync(this.net, this.meta, imageData, thresh, hier_thresh, nms);

        if (darkNetLoadedImage) {
            // memory is owned by the darknet lib
            await new Promise((res, rej) =>
                this.darknet.free_image.async(imageData, (e: any) => e ? rej(e) : res())
            );
        } else {
            // memory is owned by JS and will GC eventually
        }
        return detection;
    }

    async detectFromImageAsync(image: any, config?: IConfig): Promise<Detection[]> {
        if (!config) config = {};
        let thresh = (config.thresh) ? config.thresh : 0.5;
        let hier_thresh = (config.hier_thresh) ? config.hier_thresh : 0.5;
        let nms = (config.nms) ? config.nms : 0.5;

        return this._detectAsync(this.net, this.meta, image, thresh, hier_thresh, nms);
    }

}

export interface IConfig {
    thresh?: number;
    hier_thresh?: number;
    nms?: number;
}

export interface IBufferImage {
    b: Buffer,
    w: number,
    h: number,
    c: number
}

export type IClasses = string[]

export interface IDarknetConfig {
    weights: string;
    config: string;
    names?: string[];
    namefile?: string;
    memory?: number;
}

export interface Detection {
    name: string;
    prob: number;
    box: {
        x: number;
        y: number;
        w: number;
        h: number;
    };
}

export {Darknet} from './detector';
export {Darknet as DarknetExperimental} from './detector';
