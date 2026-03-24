#!/bin/bash
# Deploy alpha-transformer distributed collection infrastructure to AKS.
#
# Prerequisites:
#   - AKS cluster with agent-sandbox controller installed
#   - kubectl configured to point at the cluster
#   - pip install k8s-agent-sandbox
#
# Usage:
#   ./sandbox/deploy.sh              # Apply configs only
#   ./sandbox/deploy.sh --teardown   # Remove configs

set -e

if [ "$1" = "--teardown" ]; then
    echo "=== Tearing down sandbox infrastructure ==="
    kubectl delete sandboxwarmpool alpha-transformer-pool --ignore-not-found
    kubectl delete sandboxtemplate alpha-transformer-sandbox --ignore-not-found
    echo "Done."
    exit 0
fi

echo "=== Deploying alpha-transformer sandbox infrastructure ==="

# Apply sandbox template (defines what each collection pod looks like)
echo "Applying SandboxTemplate..."
kubectl apply -f sandbox/sandbox-template.yaml

# Apply warm pool (pre-creates N pods for instant dispatch)
echo "Applying WarmPool (20 replicas)..."
kubectl apply -f sandbox/warm-pool.yaml

echo ""
echo "=== Infrastructure ready ==="
echo "Warm pool: 20 pods pre-created"
echo ""
echo "To collect data:"
echo "  python collect_distributed.py --env snake --episodes 10000 --batch 50 --parallel 50 --template alpha-transformer-sandbox"
echo ""
echo "To check pod status:"
echo "  kubectl get sandboxwarmpools"
echo "  kubectl get pods -l sandbox-template=alpha-transformer-sandbox"
